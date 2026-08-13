#include "voxel_map.hpp"
#include <Eigen/Dense>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>
 
// ─────────────────────────────────────────────────────────────────────────
//  RANSAC 바닥 평면 보정
//
//  알고리즘:
//    1. 전체 포인트의 z값 정렬 → 하위 floor_z_percentile% 포인트만 추출
//       (바닥은 z값이 낮은 쪽에 있으므로 천장과 구분됨)
//    2. 추출된 포인트로 RANSAC 수행
//       → 가장 많은 inlier를 가진 평면 = 바닥 평면
//    3. 바닥 normal n_floor 추출
//    4. n_floor → [0,0,1] 로 맞추는 회전 행렬 R 계산
//       R = I + [v]× + [v]× * [v]× * (1/(1+c))
//       (Rodrigues 공식, v = n_floor × ẑ, c = n_floor · ẑ)
//    5. 전체 포인트에 R 적용 후 floor inlier 평균 z가 0이 되도록 평행이동
// ─────────────────────────────────────────────────────────────────────────
pcl::PointCloud<pcl::PointXYZ>::Ptr correctFloorAlignment(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    const ClassifyParams &params,
    Eigen::Matrix3f &out_R,
    float &out_floor_z)
{
    out_R = Eigen::Matrix3f::Identity();
    out_floor_z = 0.0f;
 
    // ── Step 1. z 하위 N% 포인트 추출 ────────────────────────────────────
    std::vector<float> z_vals;
    z_vals.reserve(cloud->size());
    for (const auto &pt : cloud->points)
        if (std::isfinite(pt.z)) z_vals.push_back(pt.z);
 
    std::sort(z_vals.begin(), z_vals.end());
    int pct_idx = static_cast<int>(z_vals.size() * params.floor_z_percentile / 100.0f);
    pct_idx = std::max(3, std::min(pct_idx, static_cast<int>(z_vals.size()) - 1));
    float z_thresh = z_vals[pct_idx];
 
    std::vector<Eigen::Vector3f> floor_pts;
    for (const auto &pt : cloud->points)
    {
        if (!std::isfinite(pt.x)||!std::isfinite(pt.y)||!std::isfinite(pt.z)) continue;
        if (pt.z <= z_thresh)
            floor_pts.emplace_back(pt.x, pt.y, pt.z);
    }
 
    std::cout << "  [floor_align] z 하위 " << params.floor_z_percentile
              << "% → z_thresh=" << z_thresh
              << "m, 바닥 후보 포인트: " << floor_pts.size() << "개\n";
 
    if (static_cast<int>(floor_pts.size()) < 3)
    {
        std::cerr << "  [floor_align] 바닥 후보 포인트 부족. "
                     "floor_z_percentile 을 높여보세요.\n";
        return cloud;
    }

    // ── Step 2. RANSAC 바닥 평면 추정 ────────────────────────────────────
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(floor_pts.size())-1);
 
    int best_inliers = 0;
    Eigen::Vector3f best_normal(0,0,1);
    float best_d = 0.0f;
    const float thresh = params.floor_ransac_thresh;
 
    for (int iter = 0; iter < params.floor_ransac_iters; ++iter)
    {
        // 3점 무작위 샘플
        int i0 = dist(rng), i1 = dist(rng), i2 = dist(rng);
        if (i0==i1 || i1==i2 || i0==i2) continue;
 
        Eigen::Vector3f p0 = floor_pts[i0];
        Eigen::Vector3f p1 = floor_pts[i1];
        Eigen::Vector3f p2 = floor_pts[i2];
 
        // 평면 normal
        Eigen::Vector3f n = (p1-p0).cross(p2-p0);
        if (n.norm() < 1e-6) continue;
        n.normalize();
 
        // n이 위(+z)를 향하도록 부호 통일
        if (n(2) < 0) n = -n;
 
        float d = -n.dot(p0);
 
        // inlier 카운트
        int cnt = 0;
        for (const auto &p : floor_pts)
            if (std::fabs(n.dot(p) + d) < thresh) cnt++;
 
        if (cnt > best_inliers)
        {
            best_inliers = cnt;
            best_normal  = n;
            best_d       = d;
        }
    }
 
    float inlier_ratio = static_cast<float>(best_inliers) /
                         static_cast<float>(floor_pts.size()) * 100.0f;
    std::cout << "  [floor_align] RANSAC 완료\n"
              << "    바닥 normal: ["
              << best_normal(0) << ", "
              << best_normal(1) << ", "
              << best_normal(2) << "]\n"
              << "    inlier 비율: " << best_inliers << " / "
              << floor_pts.size() << " (" << inlier_ratio << "%)\n";
 
    if (inlier_ratio < 30.0f)
    {
        std::cerr << "  [floor_align] inlier 비율이 너무 낮음. "
                     "floor_z_percentile 또는 floor_ransac_thresh 조정 필요.\n";
        return cloud;
    }
 
    // ── Step 3. n_floor → ẑ=[0,0,1] 로 맞추는 회전 행렬 계산 ─────────────
    Eigen::Vector3f z_axis(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f v = best_normal.cross(z_axis);  // 회전축
    float           c = best_normal.dot(z_axis);    // cos(angle)
 
    Eigen::Matrix3f R;
    if (std::fabs(c + 1.0f) < 1e-6f)
    {
        // 바닥 normal이 이미 -z: 180° 회전 (x축 기준)
        R = Eigen::Matrix3f::Identity();
        R(1,1) = -1; R(2,2) = -1;
    }
    else
    {
        // Rodrigues: R = I + [v]× + [v]×² * 1/(1+c)
        Eigen::Matrix3f vx;
        vx <<      0, -v(2),  v(1),
              v(2),     0, -v(0),
             -v(1),  v(0),     0;
        R = Eigen::Matrix3f::Identity() + vx + vx*vx*(1.0f/(1.0f+c));
    }
 
    double tilt_angle = std::acos(std::min(1.0f, std::fabs(c)))
                        * 180.0 / M_PI;
    std::cout << "  [floor_align] z축 기울기 보정 각도: "
              << tilt_angle << " deg\n";
 
    out_R = R;
 
    // RANSAC floor inlier들의 회전 후 평균 z를 보정 기준으로 쓴다.
    // 각 점을 평면에 투영하지 않고, 전체 cloud에 동일한 z offset만 적용한다.
    double floor_z_sum = 0.0;
    int floor_z_count = 0;
    for (const auto &p : floor_pts)
    {
        if (std::fabs(best_normal.dot(p) + best_d) >= thresh) continue;
        floor_z_sum += static_cast<double>((R * p).z());
        floor_z_count++;
    }
    const float floor_z_shift = (floor_z_count > 0)
        ? static_cast<float>(floor_z_sum / static_cast<double>(floor_z_count))
        : -best_d;
    out_floor_z = 0.0f;

    std::cout << "  [floor_align] floor inlier 평균 z shift: "
              << floor_z_shift << " m → 보정 후 floor z=0\n";

    // ── Step 4. 전체 포인트에 R 적용 + z 평행이동 ────────────────────────
    pcl::PointCloud<pcl::PointXYZ>::Ptr corrected(
        new pcl::PointCloud<pcl::PointXYZ>());
    corrected->reserve(cloud->size());
 
    for (const auto &pt : cloud->points)
    {
        if (!std::isfinite(pt.x)||!std::isfinite(pt.y)||!std::isfinite(pt.z))
            continue;
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        Eigen::Vector3f q = R * p;
        q(2) -= floor_z_shift;
        pcl::PointXYZ out;
        out.x = q(0); out.y = q(1); out.z = q(2);
        corrected->push_back(out);
    }
    corrected->width    = static_cast<uint32_t>(corrected->points.size());
    corrected->height   = 1;
    corrected->is_dense = true;
    return corrected;
}


// ─────────────────────────────────────────────────────────────────────────
//  [실험용] z 상위 N% 포인트 제거 (천장 없는 환경 시뮬레이션)
//
//  floor alignment 이후 호출 전제. z 상위 N% 제거 = 천장 근처 포인트 제거.
// ─────────────────────────────────────────────────────────────────────────
pcl::PointCloud<pcl::PointXYZ>::Ptr cropPointsAboveZ(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    float cut_percentile)
{
    // z 임계값 계산 (상위 cut_percentile% 자르기)
    std::vector<float> z_vals;
    z_vals.reserve(cloud->size());
    for (const auto &pt : cloud->points)
        if (std::isfinite(pt.z)) z_vals.push_back(pt.z);

    if (z_vals.size() < 3) return cloud;

    std::sort(z_vals.begin(), z_vals.end());
    int keep_idx = static_cast<int>(z_vals.size() * (100.0f - cut_percentile) / 100.0f);
    keep_idx = std::max(0, std::min(keep_idx, static_cast<int>(z_vals.size()) - 1));
    float z_cut = z_vals[keep_idx];

    // z < z_cut 인 포인트만 보존
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped(
        new pcl::PointCloud<pcl::PointXYZ>());
    cropped->reserve(static_cast<size_t>(z_vals.size() *
                                         (100.0f - cut_percentile) / 100.0f));
    for (const auto &pt : cloud->points)
    {
        if (!std::isfinite(pt.x)||!std::isfinite(pt.y)||!std::isfinite(pt.z)) continue;
        if (pt.z < z_cut) cropped->push_back(pt);
    }
    cropped->width    = static_cast<uint32_t>(cropped->points.size());
    cropped->height   = 1;
    cropped->is_dense = true;

    std::cout << "  [crop_z] z_cut = " << z_cut << " m, 제거 비율: "
              << cut_percentile << " %\n"
              << "  [crop_z] " << cloud->size() << " → " << cropped->size()
              << " 포인트\n";

    return cropped;
}

// ───────────────────────────────────────
//  Voxel Map + Pillar Map 구성
//  VoxelPoints 에 k_index 저장 (span 필터에서 사용)
// ───────────────────────────────────────
void buildVoxelAndPillarMap(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    const ClassifyParams &params,
    std::unordered_map<VoxelKey,  VoxelPoints, VoxelKeyHash>  &voxel_map,
    std::unordered_map<PillarKey, PillarData,  PillarKeyHash> &pillar_map,
    int &out_voxel_min_removed,
    BuildTiming *timing)
{
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    out_voxel_min_removed = 0;
    const float inv = 1.0f / params.voxel_size;
    if (voxel_map.empty())
        voxel_map.reserve(std::max<size_t>(1, cloud->size() / 2));

    // ── Step 1. 포인트 → voxel 할당 ──────────────────────────────────────
    // OpenMP가 없으면 스레드 수와 무관하게 직렬 경로만 쓴다.
    // (pragma가 무시되면 tid가 항상 0이라 1/T만 삽입되는 버그가 되기 때문)
#ifdef _OPENMP
    const int nthreads = std::max(1, params.voxel_num_threads);
#else
    const int nthreads = 1;
#endif

    auto t1 = Clock::now();

    if (nthreads <= 1)
    {
        for (const auto &pt : cloud->points)
        {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
                continue;

            VoxelKey key;
            key.i = static_cast<int64_t>(std::floor(pt.x * inv));
            key.j = static_cast<int64_t>(std::floor(pt.y * inv));
            key.k = static_cast<int64_t>(std::floor(pt.z * inv));

            // voxel_map[key] 조회를 1회로 제한 (기존: 2회 중복 조회)
            auto &vdat = voxel_map[key];
            vdat.k_index = key.k;
            vdat.points.emplace_back(pt.x, pt.y, pt.z);
        }
        t1 = Clock::now();
    }
    else
    {
        // 해시 분할: 스레드 t는 hash(key) % T == t 인 키만 자기 shard에 넣는다.
        // 키가 스레드별로 배타적이라 같은 shard를 두 스레드가 건드릴 일이 없고,
        // 서로 다른 shard에 같은 키가 생기지도 않는다 → 락도 중복 제거도 불필요.
        // 대가는 각 스레드가 전체 포인트를 훑어야 한다는 것(키 계산이 T배 중복).
        using VoxMap = std::unordered_map<VoxelKey, VoxelPoints, VoxelKeyHash>;
        std::vector<VoxMap> shards(nthreads);
        const size_t shard_reserve = cloud->size() / (2u * nthreads) + 1u;

        #pragma omp parallel num_threads(nthreads)
        {
            const int tid = omp_get_thread_num();
            VoxMap &m = shards[tid];
            m.reserve(shard_reserve);
            const VoxelKeyHash hasher;

            for (const auto &pt : cloud->points)
            {
                if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
                    continue;

                VoxelKey key;
                key.i = static_cast<int64_t>(std::floor(pt.x * inv));
                key.j = static_cast<int64_t>(std::floor(pt.y * inv));
                key.k = static_cast<int64_t>(std::floor(pt.z * inv));

                if (static_cast<int>(hasher(key) % static_cast<size_t>(nthreads)) != tid)
                    continue;

                auto &vdat = m[key];
                vdat.k_index = key.k;
                vdat.points.emplace_back(pt.x, pt.y, pt.z);
            }
        }
        t1 = Clock::now();

        // shard → 단일 map 병합. extract/insert는 노드를 그대로 옮기므로
        // VoxelPoints의 주소가 보존된다. PillarData.voxels가 이 주소를 들고
        // 있으므로 값 복사로 옮기면 안 된다.
        for (auto &sh : shards)
        {
            for (auto it = sh.begin(); it != sh.end(); )
            {
                auto next = std::next(it);
                voxel_map.insert(sh.extract(it));
                it = next;
            }
        }
    }

    const auto t_merge = Clock::now();

    if (pillar_map.empty())
        pillar_map.reserve(voxel_map.size());

    // ── Step 2. 유효 voxel 만 pillar에 연결 ──────────────────────────────
    //    voxel_map이 포인트를 소유하고 pillar는 포인터와 총 포인트 수만 보관한다.
    for (const auto &kv : voxel_map)
    {
        const VoxelKey    &vkey = kv.first;
        const VoxelPoints &vdat = kv.second;

        if (static_cast<int>(vdat.points.size()) < params.n_voxel_min)
        {
            out_voxel_min_removed += static_cast<int>(vdat.points.size());
            continue;
        }

        PillarKey  pkey{vkey.i, vkey.j};
        PillarData &pdat = pillar_map[pkey];
        pdat.point_count += vdat.points.size();
        pdat.voxels.push_back(&vdat);
    }

    if (timing)
    {
        using Ms = std::chrono::duration<double, std::milli>;
        timing->step1_assign_ms = Ms(t1 - t0).count();
        timing->step1_merge_ms  = Ms(t_merge - t1).count();
        timing->step2_pillar_ms = Ms(Clock::now() - t_merge).count();
    }
}
