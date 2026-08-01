#include "pillar_pca.hpp"
#include "types.hpp"
#include "voxel_map.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

namespace
{

using Clock = std::chrono::steady_clock;
using CloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using CloudRGB = pcl::PointCloud<pcl::PointXYZRGB>;

bool ensureDirectory(const std::string &dir)
{
    if (dir.empty())
        return false;

    std::string current;
    size_t start = 0;
    if (dir[0] == '/')
    {
        current = "/";
        start = 1;
    }

    while (start < dir.size())
    {
        size_t next = dir.find('/', start);
        std::string part = dir.substr(start, next - start);
        if (!part.empty())
        {
            if (current.size() > 1)
                current += "/";
            current += part;
            if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
            {
                ROS_ERROR_STREAM("mkdir failed: " << current
                                 << " (" << std::strerror(errno) << ")");
                return false;
            }
        }
        if (next == std::string::npos)
            break;
        start = next + 1;
    }
    return true;
}

bool ensureParentDirectory(const std::string &path)
{
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0)
        return true;
    return ensureDirectory(path.substr(0, slash));
}

std::string framePath(const std::string &input_dir, int index, int zero_pad)
{
    std::ostringstream name;
    name << std::setw(zero_pad) << std::setfill('0') << index << ".pcd";
    if (!input_dir.empty() && input_dir.back() == '/')
        return input_dir + name.str();
    return input_dir + "/" + name.str();
}

std::string windowName(int start_index, int end_index, int zero_pad)
{
    std::ostringstream name;
    name << std::setw(zero_pad) << std::setfill('0') << start_index
         << "_" << std::setw(zero_pad) << std::setfill('0') << end_index;
    return name.str();
}

double elapsedMs(const Clock::time_point &start, const Clock::time_point &end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

CloudXYZ::Ptr loadFrame(const std::string &path)
{
    CloudXYZ::Ptr cloud(new CloudXYZ());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) < 0)
        return CloudXYZ::Ptr();

    CloudXYZ::Ptr finite(new CloudXYZ());
    finite->reserve(cloud->size());
    for (const auto &pt : cloud->points)
    {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;
        finite->push_back(pt);
    }
    finite->width = static_cast<uint32_t>(finite->size());
    finite->height = 1;
    finite->is_dense = true;
    return finite;
}

CloudXYZ::Ptr makeAccumulatedCloud(const std::deque<CloudXYZ::Ptr> &frames)
{
    CloudXYZ::Ptr accumulated(new CloudXYZ());
    size_t total_points = 0;
    for (const auto &frame : frames)
        total_points += frame->size();
    accumulated->reserve(total_points);

    for (const auto &frame : frames)
        *accumulated += *frame;

    accumulated->width = static_cast<uint32_t>(accumulated->size());
    accumulated->height = 1;
    accumulated->is_dense = true;
    return accumulated;
}

CloudXYZ::Ptr downsampleCloud(const CloudXYZ::Ptr &cloud, float leaf_size)
{
    CloudXYZ::Ptr ds(new CloudXYZ());
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*ds);
    return ds;
}

// Planar(surface) feature 점을 voxel-filter로 다운샘플 (voxel당 centroid 1점).
// per-window 클라우드에 적용하면 bbox가 작아 작은 leaf(≤0.05m)에서도
// VoxelGrid 정수 인덱스 오버플로 없이 동작한다.
CloudRGB::Ptr downsampleSurfaces(const CloudRGB::Ptr &cloud, float leaf_size)
{
    CloudRGB::Ptr ds(new CloudRGB());
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*ds);
    ds->width = static_cast<uint32_t>(ds->size());
    ds->height = 1;
    ds->is_dense = true;
    return ds;
}

void splitFeatureClasses(const CloudRGB::Ptr &features,
                         CloudRGB::Ptr &corners,
                         CloudRGB::Ptr &surfaces)
{
    corners.reset(new CloudRGB());
    surfaces.reset(new CloudRGB());
    corners->reserve(features->size());
    surfaces->reserve(features->size());

    for (const auto &pt : features->points)
    {
        const bool is_surface =
            pt.r > 200 && pt.g > 200 && pt.b > 200;
        if (is_surface)
            surfaces->push_back(pt);
        else
            corners->push_back(pt);
    }

    corners->width = static_cast<uint32_t>(corners->size());
    corners->height = 1;
    corners->is_dense = true;
    surfaces->width = static_cast<uint32_t>(surfaces->size());
    surfaces->height = 1;
    surfaces->is_dense = true;
}

// ── corner distance-spacing 다운샘플 (pillar 단위) + 인접 anchor 초록 세그먼트 ──
// 1) corner 점을 pillar(분류 voxel_size xy 격자) + 색별로 그룹화.
// 2) anchor: 각 pillar 안에서 독립적으로 strict ≥ corner_spacing greedy 선별
//    (pillar 전용 격자로 27셀만 검사). pillar 단위라 다른 pillar의 anchor와는
//    d보다 가까울 수 있음. 실제 측정점을 남김(centroid 아님, 랜덤 아님).
// 3) edge 시각화: 같은 색 anchor 중 서로 1.5×spacing 이내인 쌍을 초록 점으로 짧게 연결.
struct CornerEdgeStats
{
    int blue_anchor = 0;  // 파랑 CORNER 선별 anchor 수
    int red_anchor = 0;   // 빨강 VOXEL_CORNER 선별 anchor 수
    int segment = 0;      // 인접 anchor 초록 세그먼트 수
};

// cell -> 그 셀에 유지된 anchor들의 flat 벡터 인덱스
using CornerGrid = std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash>;

// cell 주변 27셀에 v로부터 d 미만인 kept anchor가 없으면 true.
inline bool cornerFarEnough(const CornerGrid &grid,
                            const std::vector<Eigen::Vector3f> &pts,
                            const VoxelKey &cell, const Eigen::Vector3f &v, float d2)
{
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                const VoxelKey nk{cell.i + dx, cell.j + dy, cell.k + dz};
                const auto it = grid.find(nk);
                if (it == grid.end())
                    continue;
                for (const int idx : it->second)
                    if ((v - pts[idx]).squaredNorm() < d2)
                        return false;
            }
    return true;
}

// greedy matching: 각 anchor는 ≤1 세그먼트(=하나의 edge에만 속함).
// 미매칭 중 max_len 이내 "가장 가까운" 짝과 연결하고 초록 점 3개로 렌더.
void addGreenSegments(const CloudRGB::Ptr &out,
                      const std::vector<Eigen::Vector3f> &pts,
                      const CornerGrid &grid, double inv, float max_len,
                      int &seg_count)
{
    const float max_len2 = max_len * max_len;
    const int R = std::max(1, static_cast<int>(std::ceil(max_len * inv)));
    std::vector<char> used(pts.size(), 0);
    for (size_t i = 0; i < pts.size(); ++i)
    {
        if (used[i])
            continue;
        const Eigen::Vector3f &v = pts[i];
        const VoxelKey cell{static_cast<int64_t>(std::floor(v.x() * inv)),
                            static_cast<int64_t>(std::floor(v.y() * inv)),
                            static_cast<int64_t>(std::floor(v.z() * inv))};
        int best = -1;
        float best_d2 = max_len2;
        for (int dz = -R; dz <= R; ++dz)
            for (int dy = -R; dy <= R; ++dy)
                for (int dx = -R; dx <= R; ++dx)
                {
                    const VoxelKey nk{cell.i + dx, cell.j + dy, cell.k + dz};
                    const auto it = grid.find(nk);
                    if (it == grid.end())
                        continue;
                    for (const int j : it->second)
                    {
                        if (j == static_cast<int>(i) || used[j])
                            continue;
                        const float dd = (v - pts[j]).squaredNorm();
                        if (dd <= best_d2)
                        {
                            best_d2 = dd;
                            best = j;
                        }
                    }
                }
        if (best < 0)
            continue;
        used[i] = 1;
        used[best] = 1;
        const Eigen::Vector3f &w = pts[best];
        for (int s = 1; s <= 3; ++s)
        {
            const Eigen::Vector3f g = v + (0.25f * s) * (w - v);
            pcl::PointXYZRGB q;
            q.x = g.x();
            q.y = g.y();
            q.z = g.z();
            q.r = 0;
            q.g = 255;
            q.b = 0;
            out->push_back(q);
        }
        seg_count++;
    }
}

// 한 pillar 안에서 strict ≥ d greedy로 anchor 선별 (pillar 전용 격자로 27셀만 검사).
// pillar마다 독립이라 다른 pillar의 anchor와는 d보다 가까울 수 있다(= pillar 단위 spacing).
void spacePillarCorners(const std::vector<const pcl::PointXYZRGB *> &pillar_pts,
                        float d2, double inv, const CloudRGB::Ptr &out,
                        std::vector<Eigen::Vector3f> &anchor_pts, int &count)
{
    CornerGrid grid;                      // 이 pillar 전용
    std::vector<Eigen::Vector3f> kept;
    for (const pcl::PointXYZRGB *pp : pillar_pts)
    {
        const Eigen::Vector3f v(pp->x, pp->y, pp->z);
        const VoxelKey cell{static_cast<int64_t>(std::floor(v.x() * inv)),
                            static_cast<int64_t>(std::floor(v.y() * inv)),
                            static_cast<int64_t>(std::floor(v.z() * inv))};
        if (!cornerFarEnough(grid, kept, cell, v, d2))
            continue;
        grid[cell].push_back(static_cast<int>(kept.size()));
        kept.push_back(v);
        out->push_back(*pp);
        anchor_pts.push_back(v);
        count++;
    }
}

// 최종 anchor 점들로 전역 격자(셀=spacing) 구성 → 초록 세그먼트 인접 탐색용.
CornerGrid buildCornerGrid(const std::vector<Eigen::Vector3f> &pts, double inv)
{
    CornerGrid grid;
    for (int i = 0; i < static_cast<int>(pts.size()); ++i)
    {
        const VoxelKey cell{static_cast<int64_t>(std::floor(pts[i].x() * inv)),
                            static_cast<int64_t>(std::floor(pts[i].y() * inv)),
                            static_cast<int64_t>(std::floor(pts[i].z() * inv))};
        grid[cell].push_back(i);
    }
    return grid;
}

// corner distance-spacing 다운샘플 결과(=anchor 점만). _corner_ds.pcd 용.
CloudRGB::Ptr buildCornerAnchors(const CloudRGB::Ptr &corners,
                                 const ClassifyParams &params,
                                 CornerEdgeStats &st)
{
    CloudRGB::Ptr out(new CloudRGB());
    if (!corners || corners->empty() || params.corner_spacing <= 0.0f)
        return out;

    const float d = params.corner_spacing;
    const float d2 = d * d;
    const double inv = 1.0 / static_cast<double>(d);
    const double pinv = 1.0 / static_cast<double>(params.voxel_size);  // pillar 격자

    // 1) corner 점을 pillar(분류 voxel_size xy 격자) + 색별로 그룹화
    struct PillarCorners
    {
        std::vector<const pcl::PointXYZRGB *> blue;
        std::vector<const pcl::PointXYZRGB *> red;
    };
    std::unordered_map<PillarKey, PillarCorners, PillarKeyHash> pillars;
    for (const auto &p : corners->points)
    {
        const bool is_blue = p.b > 200 && p.r < 80 && p.g < 80;
        const bool is_red = p.r > 200 && p.g < 80 && p.b < 80;
        if (!is_blue && !is_red)
            continue;
        const PillarKey pk{static_cast<int64_t>(std::floor(p.x * pinv)),
                           static_cast<int64_t>(std::floor(p.y * pinv))};
        if (is_blue)
            pillars[pk].blue.push_back(&p);
        else
            pillars[pk].red.push_back(&p);
    }

    // 2) pillar마다 독립적으로 strict ≥ d greedy spacing → anchor만
    std::vector<Eigen::Vector3f> blue_pts, red_pts;
    out->reserve(corners->size() / 4 + 1);
    for (auto &kv : pillars)
    {
        spacePillarCorners(kv.second.blue, d2, inv, out, blue_pts, st.blue_anchor);
        spacePillarCorners(kv.second.red, d2, inv, out, red_pts, st.red_anchor);
    }

    out->width = static_cast<uint32_t>(out->size());
    out->height = 1;
    out->is_dense = true;
    return out;
}

// anchor 클라우드 복사 + 인접 anchor 초록 세그먼트. _corner_edge.pcd(시각화) 용.
CloudRGB::Ptr buildCornerViz(const CloudRGB::Ptr &anchors,
                             const ClassifyParams &params, int &seg)
{
    CloudRGB::Ptr viz(new CloudRGB(*anchors));
    if (!anchors || anchors->empty() || params.corner_spacing <= 0.0f)
        return viz;

    const double inv = 1.0 / static_cast<double>(params.corner_spacing);
    std::vector<Eigen::Vector3f> blue_pts, red_pts;
    for (const auto &p : anchors->points)
    {
        if (p.b > 200 && p.r < 80 && p.g < 80)
            blue_pts.emplace_back(p.x, p.y, p.z);
        else if (p.r > 200 && p.g < 80 && p.b < 80)
            red_pts.emplace_back(p.x, p.y, p.z);
    }
    const CornerGrid blue_grid = buildCornerGrid(blue_pts, inv);
    const CornerGrid red_grid = buildCornerGrid(red_pts, inv);
    const float max_len = 1.5f * params.corner_spacing;
    addGreenSegments(viz, blue_pts, blue_grid, inv, max_len, seg);
    addGreenSegments(viz, red_pts, red_grid, inv, max_len, seg);

    viz->width = static_cast<uint32_t>(viz->size());
    viz->height = 1;
    viz->is_dense = true;
    return viz;
}

std::string addSuffixBeforeExtension(const std::string &path,
                                     const std::string &suffix)
{
    const size_t slash = path.find_last_of('/');
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos ||
        (slash != std::string::npos && dot < slash))
    {
        return path + suffix;
    }
    return path.substr(0, dot) + suffix + path.substr(dot);
}

bool saveCloud(const std::string &path, const CloudRGB::Ptr &cloud)
{
    if (!ensureParentDirectory(path))
        return false;

    if (!cloud || cloud->empty())
    {
        std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
        if (!out.is_open())
        {
            ROS_ERROR_STREAM("[pcd_window_feature_batch] failed to open "
                             << path);
            return false;
        }
        out << "# .PCD v0.7 - Point Cloud Data file format\n"
            << "VERSION 0.7\n"
            << "FIELDS x y z rgb\n"
            << "SIZE 4 4 4 4\n"
            << "TYPE F F F F\n"
            << "COUNT 1 1 1 1\n"
            << "WIDTH 0\n"
            << "HEIGHT 1\n"
            << "VIEWPOINT 0 0 0 1 0 0 0\n"
            << "POINTS 0\n"
            << "DATA ascii\n";
        return true;
    }

    try
    {
        if (pcl::io::savePCDFileBinary(path, *cloud) == 0)
            return true;
    }
    catch (const std::exception &e)
    {
        ROS_ERROR_STREAM("[pcd_window_feature_batch] exception while saving "
                         << path << ": " << e.what());
        return false;
    }

    ROS_ERROR_STREAM("[pcd_window_feature_batch] failed to save " << path);
    return false;
}

CloudRGB::Ptr extractFeatures(const CloudXYZ::Ptr &input,
                              const ClassifyParams &params,
                              int &voxel_min_removed,
                              int &planar_count,
                              int &corner_count)
{
    CloudXYZ::Ptr cloud = input;
    float floor_z_ref = 0.0f;

    if (params.use_floor_align)
    {
        Eigen::Matrix3f floor_R = Eigen::Matrix3f::Identity();
        cloud = correctFloorAlignment(cloud, params, floor_R, floor_z_ref);
    }
    if (params.simulate_no_ceiling)
        cloud = cropPointsAboveZ(cloud, params.ceiling_cut_percentile);
    if (params.use_downsample)
        cloud = downsampleCloud(cloud, params.downsample_size);

    std::unordered_map<VoxelKey, VoxelPoints, VoxelKeyHash> voxel_map;
    std::unordered_map<PillarKey, PillarData, PillarKeyHash> pillar_map;
    buildVoxelAndPillarMap(cloud, params, voxel_map, pillar_map, voxel_min_removed);

    int voxel_dyn_cnt = 0;
    int pillar_min_cnt = 0;
    int pillar_dyn_cnt = 0;
    int span_dyn_cnt = 0;
    int fill_dyn_cnt = 0;
    int voxel_corner_cnt = 0;
    int wall_pillar_cnt = 0;
    int corner_ds_removed = 0;

    return classifyAllPillars(
        pillar_map,
        params,
        floor_z_ref,
        planar_count,
        corner_count,
        voxel_dyn_cnt,
        pillar_min_cnt,
        pillar_dyn_cnt,
        span_dyn_cnt,
        fill_dyn_cnt,
        voxel_corner_cnt,
        wall_pillar_cnt,
        corner_ds_removed);
}

void applyGisDefaults(ClassifyParams &params)
{
    params.use_downsample = 0;
    params.downsample_size = 0.05f;
    params.use_floor_align = 0;
    params.floor_z_percentile = 10.0f;
    params.floor_ransac_thresh = 0.05f;
    params.floor_ransac_iters = 1000;
    params.voxel_size = 0.2f;
    params.n_voxel_min = 3;
    params.n_pillar_min = 3;
    params.use_voxel_filter = 0;
    params.voxel_s_thresh = 0.25;
    params.simulate_no_ceiling = 0;
    params.ceiling_cut_percentile = 18.0f;
    params.use_span_filter = 1;
    params.H_min_voxel = 0;
    params.fill_ratio_thresh = 0.4;
    params.horiz_normal_thresh = 0.9;
    params.use_floor_height_filter = 0;
    params.floor_height_radius = 5;
    params.floor_height_max = 0.7;
    params.use_phi_weight = 0;
    params.phi_mean_thresh = 0.4;
    params.phi_std_thresh = 0.4;
    params.s_thresh = 0.0015;
    params.use_weight = 1;
    params.n_ref = 50.0;
    params.mu_thresh = 0.4;
    params.sigma_thresh = 0.45;
    params.use_voxel_corner = 1;
    params.corner_multiplier = 15.0;
    params.use_corner_downsample = 0;
    params.use_planar_downsample = 1;
    params.planar_leaf = 0.05f;
    params.use_corner_edge = 1;
    params.corner_spacing = 0.1f;
}

void loadParams(ros::NodeHandle &pnh, ClassifyParams &params)
{
    applyGisDefaults(params);
    pnh.param("use_downsample", params.use_downsample, params.use_downsample);
    pnh.param("downsample_size", params.downsample_size, params.downsample_size);
    pnh.param("use_floor_align", params.use_floor_align, params.use_floor_align);
    pnh.param("floor_z_percentile", params.floor_z_percentile, params.floor_z_percentile);
    pnh.param("floor_ransac_thresh", params.floor_ransac_thresh, params.floor_ransac_thresh);
    pnh.param("floor_ransac_iters", params.floor_ransac_iters, params.floor_ransac_iters);
    pnh.param("voxel_size", params.voxel_size, params.voxel_size);
    pnh.param("n_voxel_min", params.n_voxel_min, params.n_voxel_min);
    pnh.param("n_pillar_min", params.n_pillar_min, params.n_pillar_min);
    pnh.param("use_voxel_filter", params.use_voxel_filter, params.use_voxel_filter);
    pnh.param("voxel_s_thresh", params.voxel_s_thresh, params.voxel_s_thresh);
    pnh.param("simulate_no_ceiling", params.simulate_no_ceiling, params.simulate_no_ceiling);
    pnh.param("ceiling_cut_percentile", params.ceiling_cut_percentile, params.ceiling_cut_percentile);
    pnh.param("use_span_filter", params.use_span_filter, params.use_span_filter);
    pnh.param("H_min_voxel", params.H_min_voxel, params.H_min_voxel);
    pnh.param("fill_ratio_thresh", params.fill_ratio_thresh, params.fill_ratio_thresh);
    pnh.param("horiz_normal_thresh", params.horiz_normal_thresh, params.horiz_normal_thresh);
    pnh.param("use_floor_height_filter", params.use_floor_height_filter, params.use_floor_height_filter);
    pnh.param("floor_height_radius", params.floor_height_radius, params.floor_height_radius);
    pnh.param("floor_height_max", params.floor_height_max, params.floor_height_max);
    pnh.param("use_phi_weight", params.use_phi_weight, params.use_phi_weight);
    pnh.param("phi_mean_thresh", params.phi_mean_thresh, params.phi_mean_thresh);
    pnh.param("phi_std_thresh", params.phi_std_thresh, params.phi_std_thresh);
    pnh.param("s_thresh", params.s_thresh, params.s_thresh);
    pnh.param("use_weight", params.use_weight, params.use_weight);
    pnh.param("n_ref", params.n_ref, params.n_ref);
    pnh.param("mu_thresh", params.mu_thresh, params.mu_thresh);
    pnh.param("sigma_thresh", params.sigma_thresh, params.sigma_thresh);
    pnh.param("use_voxel_corner", params.use_voxel_corner, params.use_voxel_corner);
    pnh.param("corner_multiplier", params.corner_multiplier, params.corner_multiplier);
    pnh.param("use_corner_downsample", params.use_corner_downsample, params.use_corner_downsample);
    pnh.param("use_planar_downsample", params.use_planar_downsample, params.use_planar_downsample);
    pnh.param("planar_leaf", params.planar_leaf, params.planar_leaf);
    pnh.param("use_corner_edge", params.use_corner_edge, params.use_corner_edge);
    pnh.param("corner_spacing", params.corner_spacing, params.corner_spacing);
}

bool hasStandaloneArg(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg.rfind("--", 0) == 0)
            return true;
    }
    return false;
}

bool getArgValue(int argc, char **argv, const std::string &key, std::string &value)
{
    const std::string prefix = "--" + key + "=";
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg.rfind(prefix, 0) == 0)
        {
            value = arg.substr(prefix.size());
            return true;
        }
    }
    return false;
}

void applyStandaloneArgs(int argc,
                         char **argv,
                         std::string &input_dir,
                         std::string &output_dir,
                         std::string &merged_output,
                         int &start_index,
                         int &end_index,
                         int &zero_pad,
                         int &window_frames,
                         bool &save_window_features,
                         bool &save_merged_features,
                         bool &save_accumulated_windows,
                         bool &require_full_window)
{
    std::string value;
    if (getArgValue(argc, argv, "input_dir", value)) input_dir = value;
    if (getArgValue(argc, argv, "output_dir", value)) output_dir = value;
    if (getArgValue(argc, argv, "merged_output", value)) merged_output = value;
    if (getArgValue(argc, argv, "start_index", value)) start_index = std::stoi(value);
    if (getArgValue(argc, argv, "end_index", value)) end_index = std::stoi(value);
    if (getArgValue(argc, argv, "zero_pad", value)) zero_pad = std::stoi(value);
    if (getArgValue(argc, argv, "window_frames", value)) window_frames = std::stoi(value);
    if (getArgValue(argc, argv, "save_window_features", value)) save_window_features = (value != "false" && value != "0");
    if (getArgValue(argc, argv, "save_merged_features", value)) save_merged_features = (value != "false" && value != "0");
    if (getArgValue(argc, argv, "save_accumulated_windows", value)) save_accumulated_windows = (value == "true" || value == "1");
    if (getArgValue(argc, argv, "require_full_window", value)) require_full_window = (value != "false" && value != "0");
}

}  // namespace

int main(int argc, char **argv)
{
    const bool standalone = hasStandaloneArg(argc, argv);
    ros::init(argc, argv, "pcd_window_feature_batch_node",
              ros::init_options::NoSigintHandler);

    std::string input_dir =
        "/home/rain/data/ETRI_노은역/PCD_RESULT";
    std::string output_dir =
        "/home/rain/online_static_feature_ws/output/pcd_window_features/ETRI_노은역/window_1";
    std::string merged_output;
    int start_index = 0;
    int end_index = 1832;
    int zero_pad = 5;
    int window_frames = 1;
    bool save_window_features = true;
    bool save_merged_features = true;
    bool save_accumulated_windows = false;
    bool require_full_window = true;
    std::unique_ptr<ros::NodeHandle> pnh;

    ClassifyParams params;
    if (standalone)
    {
        applyGisDefaults(params);
        applyStandaloneArgs(argc, argv, input_dir, output_dir, merged_output,
                            start_index, end_index, zero_pad, window_frames,
                            save_window_features, save_merged_features,
                            save_accumulated_windows, require_full_window);
    }
    else
    {
        pnh.reset(new ros::NodeHandle("~"));
        pnh->param<std::string>("input_dir", input_dir, input_dir);
        pnh->param<std::string>("output_dir", output_dir, output_dir);
        pnh->param<std::string>("merged_output", merged_output, merged_output);
        pnh->param("start_index", start_index, start_index);
        pnh->param("end_index", end_index, end_index);
        pnh->param("zero_pad", zero_pad, zero_pad);
        pnh->param("window_frames", window_frames, window_frames);
        pnh->param("save_window_features", save_window_features, save_window_features);
        pnh->param("save_merged_features", save_merged_features, save_merged_features);
        pnh->param("save_accumulated_windows", save_accumulated_windows, save_accumulated_windows);
        pnh->param("require_full_window", require_full_window, require_full_window);
        loadParams(*pnh, params);
    }

    window_frames = std::max(1, window_frames);
    if (merged_output.empty())
    {
        std::ostringstream out;
        out << output_dir << "/merged/merged_static_features_window_"
            << window_frames << ".pcd";
        merged_output = out.str();
    }

    if (!ensureDirectory(output_dir))
        return 1;

    ROS_INFO_STREAM("[pcd_window_feature_batch] input_dir=" << input_dir
                    << ", output_dir=" << output_dir
                    << ", window_frames=" << window_frames
                    << ", range=[" << start_index << ", " << end_index << "]");

    std::deque<CloudXYZ::Ptr> frame_buffer;
    CloudRGB::Ptr merged(new CloudRGB());
    CloudRGB::Ptr merged_corners(new CloudRGB());
    CloudRGB::Ptr merged_surfaces(new CloudRGB());
    CloudRGB::Ptr merged_surfaces_ds(new CloudRGB());
    CloudRGB::Ptr merged_corner_ds(new CloudRGB());
    CloudRGB::Ptr merged_corner_edge(new CloudRGB());
    size_t processed_windows = 0;
    size_t failed_frames = 0;
    const auto total_start = Clock::now();

    for (int index = start_index;
         index <= end_index && (standalone || ros::ok());
         ++index)
    {
        const std::string path = framePath(input_dir, index, zero_pad);
        CloudXYZ::Ptr frame = loadFrame(path);
        if (!frame)
        {
            ROS_ERROR_STREAM("[pcd_window_feature_batch] failed to load " << path);
            failed_frames++;
            continue;
        }

        frame_buffer.push_back(frame);
        while (static_cast<int>(frame_buffer.size()) > window_frames)
            frame_buffer.pop_front();

        if (require_full_window &&
            static_cast<int>(frame_buffer.size()) < window_frames)
            continue;

        const int window_start = index - static_cast<int>(frame_buffer.size()) + 1;
        const int window_end = index;
        const std::string name = windowName(window_start, window_end, zero_pad);

        const auto window_start_time = Clock::now();
        CloudXYZ::Ptr accumulated = makeAccumulatedCloud(frame_buffer);

        if (save_accumulated_windows)
        {
            const std::string accumulated_path =
                output_dir + "/" + name + "_accumulated.pcd";
            pcl::io::savePCDFileBinary(accumulated_path, *accumulated);
        }

        int voxel_min_removed = 0;
        int planar_count = 0;
        int corner_count = 0;
        CloudRGB::Ptr features = extractFeatures(
            accumulated, params, voxel_min_removed, planar_count, corner_count);
        CloudRGB::Ptr corners;
        CloudRGB::Ptr surfaces;
        splitFeatureClasses(features, corners, surfaces);

        // planar(surface) 점만 voxel-filter 다운샘플 (KD-Tree 입력용).
        // corner는 그대로 둔다. per-window 적용이라 작은 leaf도 오버플로 없음.
        CloudRGB::Ptr surfaces_ds = surfaces;
        if (params.use_planar_downsample)
            surfaces_ds = downsampleSurfaces(surfaces, params.planar_leaf);

        // corner → edge primitive 시각화 (파랑 anchor + 빨강 직선).
        CloudRGB::Ptr corner_ds, corner_edge;
        CornerEdgeStats corner_edge_stats;
        if (params.use_corner_edge)
        {
            corner_ds = buildCornerAnchors(corners, params, corner_edge_stats);
            corner_edge = buildCornerViz(corner_ds, params, corner_edge_stats.segment);
        }

        if (save_window_features)
        {
            const std::string feature_path =
                output_dir + "/" + name + "_features.pcd";
            const std::string corner_path =
                output_dir + "/" + name + "_features_corner.pcd";
            const std::string surface_path =
                output_dir + "/" + name + "_features_surface.pcd";
            if (!saveCloud(feature_path, features) ||
                !saveCloud(corner_path, corners) ||
                !saveCloud(surface_path, surfaces))
                return 1;
            if (params.use_planar_downsample)
            {
                const std::string surface_ds_path =
                    output_dir + "/" + name + "_features_surface_ds.pcd";
                if (!saveCloud(surface_ds_path, surfaces_ds))
                    return 1;
            }
            if (params.use_corner_edge)
            {
                const std::string corner_ds_path =
                    output_dir + "/" + name + "_features_corner_ds.pcd";
                const std::string corner_edge_path =
                    output_dir + "/" + name + "_features_corner_edge.pcd";
                if (!saveCloud(corner_ds_path, corner_ds) ||
                    !saveCloud(corner_edge_path, corner_edge))
                    return 1;
            }
        }

        *merged += *features;
        *merged_corners += *corners;
        *merged_surfaces += *surfaces;
        *merged_surfaces_ds += *surfaces_ds;
        // merged corner_edge는 루프 후 누적 corner에 전역 spacing 1회로 dedup.
        processed_windows++;

        const double window_ms = elapsedMs(window_start_time, Clock::now());
        ROS_INFO_STREAM("[pcd_window_feature_batch] " << name
                        << " accumulated=" << accumulated->size()
                        << " features=" << features->size()
                        << " surface=" << surfaces->size()
                        << " surface_ds=" << surfaces_ds->size()
                        << " corner=" << corners->size()
                        << " corner_ds=" << (corner_ds ? corner_ds->size() : 0)
                        << " corner_edge=" << (corner_edge ? corner_edge->size() : 0)
                        << " corner_seg=" << corner_edge_stats.segment
                        << " planar_label_count=" << planar_count
                        << " corner_label_count=" << corner_count
                        << " voxel_min_removed=" << voxel_min_removed
                        << " time_ms=" << window_ms);
    }

    merged->width = static_cast<uint32_t>(merged->size());
    merged->height = 1;
    merged->is_dense = true;
    merged_corners->width = static_cast<uint32_t>(merged_corners->size());
    merged_corners->height = 1;
    merged_corners->is_dense = true;
    merged_surfaces->width = static_cast<uint32_t>(merged_surfaces->size());
    merged_surfaces->height = 1;
    merged_surfaces->is_dense = true;
    merged_surfaces_ds->width = static_cast<uint32_t>(merged_surfaces_ds->size());
    merged_surfaces_ds->height = 1;
    merged_surfaces_ds->is_dense = true;
    // merged corner_ds: 누적 corner 전체를 pillar 단위 distance-spacing 1회 → window 간 중복 제거.
    if (params.use_corner_edge)
    {
        CornerEdgeStats merged_stats;
        merged_corner_ds = buildCornerAnchors(merged_corners, params, merged_stats);
        merged_corner_edge = buildCornerViz(merged_corner_ds, params, merged_stats.segment);
        ROS_INFO_STREAM("[pcd_window_feature_batch] merged corner_ds: "
                        << merged_corners->size() << " -> " << merged_corner_ds->size()
                        << " (blue=" << merged_stats.blue_anchor
                        << ", red=" << merged_stats.red_anchor
                        << ", seg=" << merged_stats.segment << ")");
    }
    merged_corner_ds->width = static_cast<uint32_t>(merged_corner_ds->size());
    merged_corner_ds->height = 1;
    merged_corner_ds->is_dense = true;
    merged_corner_edge->width = static_cast<uint32_t>(merged_corner_edge->size());
    merged_corner_edge->height = 1;
    merged_corner_edge->is_dense = true;

    if (save_merged_features)
    {
        const std::string merged_corner_output =
            addSuffixBeforeExtension(merged_output, "_corner");
        const std::string merged_surface_output =
            addSuffixBeforeExtension(merged_output, "_surface");
        if (!saveCloud(merged_output, merged) ||
            !saveCloud(merged_corner_output, merged_corners) ||
            !saveCloud(merged_surface_output, merged_surfaces))
            return 1;
        // merged _surface_ds 는 per-window 다운샘플 결과의 단순 합집합이라
        // 윈도우 겹침 영역에 중복이 남을 수 있음(전역 dedup 아님). 비교/검수용.
        if (params.use_planar_downsample)
        {
            const std::string merged_surface_ds_output =
                addSuffixBeforeExtension(merged_output, "_surface_ds");
            if (!saveCloud(merged_surface_ds_output, merged_surfaces_ds))
                return 1;
        }
        if (params.use_corner_edge)
        {
            const std::string merged_corner_ds_output =
                addSuffixBeforeExtension(merged_output, "_corner_ds");
            if (!saveCloud(merged_corner_ds_output, merged_corner_ds))
                return 1;
            const std::string merged_corner_edge_output =
                addSuffixBeforeExtension(merged_output, "_corner_edge");
            if (!saveCloud(merged_corner_edge_output, merged_corner_edge))
                return 1;
        }
    }

    ROS_INFO_STREAM("[pcd_window_feature_batch] done. processed_windows="
                    << processed_windows
                    << ", failed_frames=" << failed_frames
                    << ", merged_points=" << merged->size()
                    << ", merged_corner_points=" << merged_corners->size()
                    << ", merged_surface_points=" << merged_surfaces->size()
                    << ", merged_surface_ds_points=" << merged_surfaces_ds->size()
                    << ", merged_corner_edge_points=" << merged_corner_edge->size()
                    << ", merged_output=" << merged_output
                    << ", total_ms=" << elapsedMs(total_start, Clock::now()));
    return failed_frames == 0 ? 0 : 2;
}
