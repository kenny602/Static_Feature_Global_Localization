#include <geometry_msgs/PoseStamped.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <small_gicp/registration/registration_helper.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>
#include <string>

#include <sys/stat.h>
#include <sys/types.h>

namespace
{

using PointXYZ = pcl::PointXYZ;
using PointXYZI = pcl::PointXYZI;
using CloudXYZ = pcl::PointCloud<PointXYZ>;
using CloudXYZI = pcl::PointCloud<PointXYZI>;

// 노드 내부 pose표현인 Matrix4f를 small-GICP가 사용하는 Isometry3d로 변환
Eigen::Isometry3d matrixToIsometry(const Eigen::Matrix4f &m)
{
    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.matrix() = m.cast<double>();
    return iso;
}

// small-GICP 결과를 노드 내부 pose표현으로 되돌림
Eigen::Matrix4f isometryToMatrix(const Eigen::Isometry3d &iso)
{
    return iso.matrix().cast<float>();
}

// pcl 포인트를 small-GICP가 받는 homogeneous 좌표 벡터로 변환
std::vector<Eigen::Vector4f> cloudToEigenPoints(const CloudXYZ::Ptr &cloud)
{
    std::vector<Eigen::Vector4f> points;
    points.reserve(cloud->size());
    for (const auto &p : cloud->points)
        points.emplace_back(p.x, p.y, p.z, 1.0f); // 1은 translation 적용을 위함
    return points;
}

// 문자열 파라미터(정합 모드)를 small-GICP enum으로 변환함
small_gicp::RegistrationSetting::RegistrationType parseRegistrationType(
    const std::string &type)
{
    std::string lower = type;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower == "icp")
        return small_gicp::RegistrationSetting::ICP;
    if (lower == "plane_icp" || lower == "plane-icp" || lower == "planeicp")
        return small_gicp::RegistrationSetting::PLANE_ICP;
    if (lower == "vgicp")
        return small_gicp::RegistrationSetting::VGICP;
    return small_gicp::RegistrationSetting::GICP;
}

// 4x4 pose 행렬을 ROS의 PoseStamped로 변환
geometry_msgs::PoseStamped poseMsgFromMatrix(const Eigen::Matrix4f &T,
                                             const ros::Time &stamp,
                                             const std::string &frame_id)
{
    const Eigen::Matrix3f R = T.block<3, 3>(0, 0);
    const Eigen::Quaternionf q(R);

    geometry_msgs::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = frame_id;
    pose.pose.position.x = T(0, 3);
    pose.pose.position.y = T(1, 3);
    pose.pose.position.z = T(2, 3);
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();
    pose.pose.orientation.w = q.w();
    return pose;
}

}  // namespace

class ScanMatchingAccumulatorNode
{
public:
    ScanMatchingAccumulatorNode()
        // ros::NodeHandle: ROS 시스템과 연결되는 여러 리소스를 만드는 진입점
        : nh_(),    // nh  = 일반 NodeHandle, namespace: 일반/현재 ROS namespace, 외부 토픽 subscribe
          pnh_("~") // png = private NodeHandle (현재 노드의 private namespace를 사용하는 NodeHandle), 노드 전용 파라미터와 출력 토픽용
    {
        loadParams();
        initLatencyLog();

        cloud_sub_ = nh_.subscribe(input_topic_, 2,
                                   &ScanMatchingAccumulatorNode::cloudCallback,
                                   this);
        dense_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            dense_window_topic_, 1, false);
        submap_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            submap_topic_, 1, false);
        pose_pub_ = pnh_.advertise<geometry_msgs::PoseStamped>(
            pose_topic_, 20, false);

        ROS_INFO_STREAM("[scan_matching_accumulator] input_topic=" << input_topic_
                        << ", window_frames=" << window_frames_
                        << ", submap_frames=" << submap_frames_
                        << ", registration_type=" << registration_type_
                        << ", odom_frame=" << odom_frame_);
    }

private:
    using SteadyClock = std::chrono::steady_clock;
    using TimePoint = SteadyClock::time_point;

    struct Frame
    {
        ros::Time stamp;
        std::string lidar_frame;
        CloudXYZI::Ptr raw;
        Eigen::Matrix4f T_odom_lidar = Eigen::Matrix4f::Identity();
    };

    struct LatencyRecord
    {
        int callback_index = 0;
        int window_size = 0;
        int dense_published = 0;
        double msg_stamp_sec = 0.0;
        double wall_stamp_sec = 0.0;

        size_t raw_msg_points = 0;
        size_t filtered_points = 0;
        size_t registration_source_points = 0;
        size_t registration_target_points = 0;
        size_t dense_points = 0;

        int registration_converged = 0;
        size_t registration_inliers = 0;
        double registration_error = 0.0;

        double preprocess_ms = 0.0;
        double submap_build_ms = 0.0;
        double source_prepare_ms = 0.0;
        double small_gicp_prepare_ms = 0.0;
        double small_gicp_align_ms = 0.0;
        double buffer_ms = 0.0;
        double pose_publish_ms = 0.0;
        double dense_window_build_ms = 0.0;
        double dense_window_publish_ms = 0.0;
        double submap_publish_ms = 0.0;
        double callback_total_ms = 0.0;
    };

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber cloud_sub_;
    ros::Publisher dense_pub_;
    ros::Publisher submap_pub_;
    ros::Publisher pose_pub_;

    std::string input_topic_ = "/ouster/points";
    std::string odom_frame_ = "odom";
    std::string dense_window_topic_ = "dense_window_cloud";
    std::string submap_topic_ = "registration_submap";
    std::string pose_topic_ = "scan_matching_pose";
    std::string registration_type_ = "VGICP";
    std::string latency_log_path_ =
        "/home/rain/online_static_feature_ws/output/bag_scan_matching/scan_matching_accumulator_latency.csv";

    int window_frames_ = 4;
    int submap_frames_ = 6;
    int num_threads_ = 4;
    int correspondence_randomness_ = 20;
    int max_iterations_ = 32;
    int publish_every_n_frames_ = 1;
    int min_registration_points_ = 200;

    double min_range_ = 1.0;
    double max_range_ = 120.0;
    double z_min_ = -3.0;
    double z_max_ = 8.0;
    double registration_leaf_size_ = 0.5;
    double dense_leaf_size_ = 0.0;
    double max_correspondence_distance_ = 2.0;
    double voxel_resolution_ = 1.0;
    double transformation_epsilon_ = 1e-4;
    double rotation_epsilon_ = 1e-4;

    bool use_constant_velocity_guess_ = true;
    bool publish_submap_ = false;
    bool publish_only_full_window_ = true;
    bool latency_log_enabled_ = true;
    bool verbose_registration_ = false;

    int callback_count_ = 0;
    std::deque<Frame> frames_;
    Eigen::Matrix4f last_delta_ = Eigen::Matrix4f::Identity();
    // last_delta_가 "직전 두 프레임의 정합 결과"로 채워졌는지 여부.
    // frames_ 버퍼 크기(max(window, submap))와 무관하게 판단해야 한다.
    // window=submap=1이면 버퍼가 항상 1장이라 frames_.size()>=2 조건으로는
    // 등속도 예측이 한 번도 켜지지 않는다.
    bool has_last_delta_ = false;
    std::ofstream latency_log_;

    void loadParams()
    {
        // pnh_.param("파라미터 이름", 저장할 변수, 기본값)
        // 1. ROS parameter server에서 private 파라미터를 찾음 (launch나 yaml 등)
        // 2. 존재하면 해당 값을 변수에 저장
        // 3. 없으면 세 번째 인자의 기본값을 사용
        pnh_.param<std::string>("input_topic", input_topic_, input_topic_);
        pnh_.param<std::string>("odom_frame", odom_frame_, odom_frame_);
        pnh_.param<std::string>("dense_window_topic", dense_window_topic_,
                                dense_window_topic_);
        pnh_.param<std::string>("submap_topic", submap_topic_, submap_topic_);
        pnh_.param<std::string>("pose_topic", pose_topic_, pose_topic_);
        pnh_.param<std::string>("registration_type", registration_type_,
                                registration_type_);
        pnh_.param<std::string>("latency_log_path", latency_log_path_,
                                latency_log_path_);

        pnh_.param("window_frames", window_frames_, window_frames_);
        pnh_.param("submap_frames", submap_frames_, submap_frames_);
        pnh_.param("num_threads", num_threads_, num_threads_);
        pnh_.param("correspondence_randomness", correspondence_randomness_,
                   correspondence_randomness_);
        pnh_.param("max_iterations", max_iterations_, max_iterations_);
        pnh_.param("publish_every_n_frames", publish_every_n_frames_,
                   publish_every_n_frames_);
        pnh_.param("min_registration_points", min_registration_points_,
                   min_registration_points_);

        pnh_.param("min_range", min_range_, min_range_);
        pnh_.param("max_range", max_range_, max_range_);
        pnh_.param("z_min", z_min_, z_min_);
        pnh_.param("z_max", z_max_, z_max_);
        pnh_.param("registration_leaf_size", registration_leaf_size_,
                   registration_leaf_size_);
        pnh_.param("dense_leaf_size", dense_leaf_size_, dense_leaf_size_);
        pnh_.param("max_correspondence_distance", max_correspondence_distance_,
                   max_correspondence_distance_);
        pnh_.param("voxel_resolution", voxel_resolution_, voxel_resolution_);
        pnh_.param("transformation_epsilon", transformation_epsilon_,
                   transformation_epsilon_);
        pnh_.param("rotation_epsilon", rotation_epsilon_, rotation_epsilon_);

        pnh_.param("use_constant_velocity_guess", use_constant_velocity_guess_,
                   use_constant_velocity_guess_);
        pnh_.param("publish_submap", publish_submap_, publish_submap_);
        pnh_.param("publish_only_full_window", publish_only_full_window_,
                   publish_only_full_window_);
        pnh_.param("latency_log_enabled", latency_log_enabled_,
                   latency_log_enabled_);
        pnh_.param("verbose_registration", verbose_registration_,
                   verbose_registration_);

        window_frames_ = std::max(1, window_frames_);
        submap_frames_ = std::max(1, submap_frames_);
        num_threads_ = std::max(1, num_threads_);
        correspondence_randomness_ = std::max(3, correspondence_randomness_);
        max_iterations_ = std::max(1, max_iterations_);
        publish_every_n_frames_ = std::max(1, publish_every_n_frames_);
        min_registration_points_ = std::max(10, min_registration_points_);
    }

    // 두 시점 사이를 ms로 변환
    static double elapsedMs(const TimePoint &start, const TimePoint &end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    // 실시간성 분석을 위한 latency CSV를 저장할 디렉토리 생성 
    bool ensureParentDirectory(const std::string &path) const
    {
        const size_t slash = path.find_last_of('/');
        if (slash == std::string::npos || slash == 0)
            return true;

        std::string current;
        const std::string dir = path.substr(0, slash);
        size_t start = 0;
        if (!dir.empty() && dir[0] == '/')
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
                    ROS_ERROR_STREAM("[scan_matching_accumulator] mkdir failed: "
                                     << current << " (" << std::strerror(errno)
                                     << ")");
                    return false;
                }
            }
            if (next == std::string::npos)
                break;
            start = next + 1;
        }
        return true;
    }

    // CSV 파일을 열고 헤더 기록
    void initLatencyLog()
    {
        if (!latency_log_enabled_)
            return;

        if (!ensureParentDirectory(latency_log_path_))
        {
            latency_log_enabled_ = false;
            return;
        }

        latency_log_.open(latency_log_path_.c_str(), std::ios::out | std::ios::trunc);
        if (!latency_log_.is_open())
        {
            ROS_ERROR_STREAM("[scan_matching_accumulator] failed to open latency log: "
                             << latency_log_path_);
            latency_log_enabled_ = false;
            return;
        }

        latency_log_
            << "callback_index,window_size,dense_published,msg_stamp_sec,wall_stamp_sec,"
            << "raw_msg_points,filtered_points,registration_source_points,"
            << "registration_target_points,dense_points,"
            << "registration_converged,registration_inliers,registration_error,"
            << "preprocess_ms,submap_build_ms,source_prepare_ms,"
            << "small_gicp_prepare_ms,small_gicp_align_ms,buffer_ms,"
            << "pose_publish_ms,dense_window_build_ms,"
            << "dense_window_publish_ms,submap_publish_ms,"
            << "callback_total_ms\n";
        latency_log_.flush();

        ROS_INFO_STREAM("[scan_matching_accumulator] latency log: "
                        << latency_log_path_);
    }

    // 한 callback의 LatencyRecord를 CSV 한 줄로 저장함
    void writeLatencyLog(const LatencyRecord &r)
    {
        if (!latency_log_enabled_ || !latency_log_.is_open())
            return;

        latency_log_ << std::fixed << std::setprecision(6)
            << r.callback_index << ','
            << r.window_size << ','
            << r.dense_published << ','
            << r.msg_stamp_sec << ','
            << r.wall_stamp_sec << ','
            << r.raw_msg_points << ','
            << r.filtered_points << ','
            << r.registration_source_points << ','
            << r.registration_target_points << ','
            << r.dense_points << ','
            << r.registration_converged << ','
            << r.registration_inliers << ','
            << r.registration_error << ','
            << r.preprocess_ms << ','
            << r.submap_build_ms << ','
            << r.source_prepare_ms << ','
            << r.small_gicp_prepare_ms << ','
            << r.small_gicp_align_ms << ','
            << r.buffer_ms << ','
            << r.pose_publish_ms << ','
            << r.dense_window_build_ms << ','
            << r.dense_window_publish_ms << ','
            << r.submap_publish_ms << ','
            << r.callback_total_ms << '\n';
        latency_log_.flush();
    }

    // NaN 제거, small-GICP의 거리 및 높이 필터
    // PointCloud2 -> PointXYZI 클라우드로 만듦
    CloudXYZI::Ptr preprocessRawCloud(const sensor_msgs::PointCloud2ConstPtr &msg) const
    {
        CloudXYZI::Ptr in(new CloudXYZI());
        pcl::fromROSMsg(*msg, *in); // ROS 메세지를 PCL로 변환

        CloudXYZI::Ptr out(new CloudXYZI());
        out->reserve(in->size());
        const double min_r2 = min_range_ * min_range_;
        const double max_r2 = max_range_ * max_range_;

        for (const auto &p : in->points)
        {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
                !std::isfinite(p.z))
            {
                continue;
            }
            if (p.z < z_min_ || p.z > z_max_)
                continue;

            const double r2 = static_cast<double>(p.x) * p.x +
                              static_cast<double>(p.y) * p.y +
                              static_cast<double>(p.z) * p.z;
            if (r2 < min_r2 || r2 > max_r2)
                continue;

            out->push_back(p);
        }

        out->width = static_cast<uint32_t>(out->size());
        out->height = 1;
        out->is_dense = true;
        return out;
    }

    // PointXYZI에서 PointXYZ로 바꿈 (스캔 매칭에서는 위치 정보만 필요하기 때문)
    CloudXYZ::Ptr toXYZ(const CloudXYZI::Ptr &cloud) const
    {
        CloudXYZ::Ptr xyz(new CloudXYZ());
        xyz->reserve(cloud->size());
        for (const auto &p : cloud->points)
        {
            PointXYZ q;
            q.x = p.x;
            q.y = p.y;
            q.z = p.z;
            xyz->push_back(q);
        }
        xyz->width = static_cast<uint32_t>(xyz->size());
        xyz->height = 1;
        xyz->is_dense = true;
        return xyz;
    }

    template <typename CloudT>
    // PCL VoxelGrid 적용해서 다운샘플링 수행
    typename CloudT::Ptr voxelDownsample(const typename CloudT::Ptr &cloud,
                                         double leaf_size) const
    {
        if (leaf_size <= 1e-6 || cloud->empty())
            return cloud;

        typename CloudT::Ptr filtered(new CloudT());
        pcl::VoxelGrid<typename CloudT::PointType> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(static_cast<float>(leaf_size),
                       static_cast<float>(leaf_size),
                       static_cast<float>(leaf_size));
        vg.filter(*filtered);
        filtered->is_dense = true;
        return filtered;
    }

    // raw LiDAR cloud를 T_odom_lidar로 변환함
    CloudXYZI::Ptr transformCloud(const CloudXYZI::Ptr &cloud,
                                  const Eigen::Matrix4f &T) const
    {
        CloudXYZI::Ptr transformed(new CloudXYZI());
        // pcl 라이브러리 -> 입력, 출력, 변환 pose
        // 입력: LiDAR 좌표계 포인트, 출력: odom 좌표계 포인트
        pcl::transformPointCloud(*cloud, *transformed, T);
        // transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
        //                pcl::PointCloud<PointT> &cloud_out, 
        //                const Eigen::Vector3f &offset, 
        //                const Eigen::Quaternionf &rotation,
        //                bool copy_all_fields = true)
        transformed->width = static_cast<uint32_t>(transformed->size());
        transformed->height = 1;
        transformed->is_dense = true;
        return transformed;
    }

    // 스캔 매칭 함수
    CloudXYZ::Ptr buildRegistrationSubmap() const
    {
        CloudXYZI::Ptr submap_xyzi(new CloudXYZI());
        const int count = std::min<int>(submap_frames_, frames_.size());
        if (count <= 0)
            return CloudXYZ::Ptr(new CloudXYZ());

        size_t total_points = 0;
        // frames_에 저장된 최근 과거 프레임들로 정합 target을 만듦
        for (int i = static_cast<int>(frames_.size()) - count;
             i < static_cast<int>(frames_.size()); ++i)
        {
            total_points += frames_[i].raw->size();
        }
        submap_xyzi->reserve(total_points);

        for (int i = static_cast<int>(frames_.size()) - count;
             i < static_cast<int>(frames_.size()); ++i)
        {
            // 각 과거 프레임은 아직 자신의 LiDAR 좌표계에 저장되어 있으므로 T_odom_lidar로 변환 필요
            CloudXYZI::Ptr aligned = transformCloud(frames_[i].raw,
                                                    frames_[i].T_odom_lidar);
            *submap_xyzi += *aligned; // odom 좌표계로 옮긴 프레임들을 모두 submap에 합침
        }

        return toXYZ(submap_xyzi);
    }

    // 1. 과거 frame으로 target submap 생성
    // 2. 현재 raw를 source로 변환
    // 3. 이전 pose로 initial guess 생성
    // 4. small-GICP 정합
    Eigen::Matrix4f estimatePose(const CloudXYZI::Ptr &raw,
                                 const ros::Time &stamp,
                                 LatencyRecord &latency)
    {
        if (frames_.empty()) // 과거 프레임이 없으면
            return Eigen::Matrix4f::Identity(); // 첫 LiDAR 위치를 odom 원점으로 정의

        TimePoint stage_start = SteadyClock::now();
        CloudXYZ::Ptr target = buildRegistrationSubmap(); // target 구성 -> 과거 프레임들을 odom 좌표계로 합친 submap
        TimePoint stage_end = SteadyClock::now();
        latency.submap_build_ms = elapsedMs(stage_start, stage_end);

        stage_start = SteadyClock::now();
        CloudXYZ::Ptr source = toXYZ(raw); // source 구성 -> 현재 프레임은 아직 LiDAR 좌표계에 있음
        stage_end = SteadyClock::now();
        latency.source_prepare_ms = elapsedMs(stage_start, stage_end);
        latency.registration_target_points = target->size();
        latency.registration_source_points = source->size();

        if (target->size() < static_cast<size_t>(min_registration_points_) ||
            source->size() < static_cast<size_t>(min_registration_points_))
        {
            ROS_WARN_STREAM_THROTTLE(
                1.0,
                "[scan_matching_accumulator] too few registration points. "
                << "target=" << target->size()
                << ", source=" << source->size()
                << ". Reusing previous pose.");
            has_last_delta_ = false; // 이 프레임은 이동량이 없으므로 다음 프레임 예측에 직전 delta를 쓰면 안 된다
            return frames_.back().T_odom_lidar; // 정합에 필요한 포인트 수가 너무 적으면, 직전 pose 재사용
        }

        Eigen::Matrix4f guess = frames_.back().T_odom_lidar;     // Initial guess 기본값 - 직전 pose
        if (use_constant_velocity_guess_ && has_last_delta_)     // 등속도 모델: 직전 정합에서 구한 이동량이 있으면
            guess = last_delta_ * frames_.back().T_odom_lidar;   // last_delta_: 직전 두 pose 간 변화량 (직전 이동량이 다음 프레임에도 반복 가정)

        // 매 프레임마다 RegistrationSetting을 구성함
        small_gicp::RegistrationSetting setting;
        setting.type = parseRegistrationType(registration_type_);
        setting.num_threads = num_threads_;
        setting.downsampling_resolution = registration_leaf_size_; // small_gicp가 registration을 위해 내부적으로 사용할 downsampling 크기
        setting.max_correspondence_distance = max_correspondence_distance_;
        setting.voxel_resolution = voxel_resolution_;
        setting.max_iterations = max_iterations_;
        setting.translation_eps = transformation_epsilon_;
        setting.rotation_eps = rotation_epsilon_;
        setting.verbose = verbose_registration_;

        stage_start = SteadyClock::now();
        const std::vector<Eigen::Vector4f> target_points =
            cloudToEigenPoints(target);
        const std::vector<Eigen::Vector4f> source_points =
            cloudToEigenPoints(source);
        stage_end = SteadyClock::now();
        latency.small_gicp_prepare_ms = elapsedMs(stage_start, stage_end);

        // small-GICP 정합 시작
        stage_start = SteadyClock::now();
        const small_gicp::RegistrationResult result =
            small_gicp::align(target_points, source_points,
                              matrixToIsometry(guess), setting);
        stage_end = SteadyClock::now();

        latency.small_gicp_align_ms = elapsedMs(stage_start, stage_end);
        latency.registration_converged = result.converged ? 1 : 0;
        latency.registration_inliers = result.num_inliers;
        latency.registration_error = result.error;

        const Eigen::Matrix4f final_pose =
            isometryToMatrix(result.T_target_source); // result.T_target_source가 바로 T_odom_lidar
        ROS_INFO_STREAM_THROTTLE(
            1.0,
            "[scan_matching_accumulator] stamp=" << stamp
            << ", converged=" << result.converged
            << ", inliers=" << result.num_inliers
            << ", error=" << result.error
            << ", source=" << source->size()
            << ", target=" << target->size());

        last_delta_ = final_pose * frames_.back().T_odom_lidar.inverse();
        has_last_delta_ = true;
        return final_pose;
    }

    // DenseWindow 만들기
    CloudXYZI::Ptr buildDenseWindow() const
    {
        CloudXYZI::Ptr dense(new CloudXYZI());
        // 최근 window_frames_개의 raw frame들을 하나의 window로 합침
        const int count = std::min<int>(window_frames_, frames_.size());
        if (count <= 0)
            return dense;

        size_t total_points = 0;
        for (int i = static_cast<int>(frames_.size()) - count;
             i < static_cast<int>(frames_.size()); ++i)
        {
            total_points += frames_[i].raw->size();
        }
        dense->reserve(total_points);

        // 각 프레임은 자신의 센서 좌표계에 있으므로, 각자 다른 pose로 odom 좌표계로 옮김 
        for (int i = static_cast<int>(frames_.size()) - count;
             i < static_cast<int>(frames_.size()); ++i)
        {
            CloudXYZI::Ptr aligned = transformCloud(frames_[i].raw,
                                                    frames_[i].T_odom_lidar);
            *dense += *aligned;
        }

        dense->width = static_cast<uint32_t>(dense->size());
        dense->height = 1;
        dense->is_dense = true;

        // dense_leaf_size_가 0보다 크면 VoxelGrid를 통한 다운샘플링을 적용함
        return voxelDownsample<CloudXYZI>(dense, dense_leaf_size_);
    }

    // PCL PointXYZI 를 ROS PointCloud2로 바꿔서 발행함
    void publishCloud(const CloudXYZI::Ptr &cloud,
                      const ros::Publisher &pub,
                      const ros::Time &stamp,
                      const std::string &frame_id) const
    {
        if (!pub)
            return;

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg); // ROS로 바꿔서 발행
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id; // 모든 포인트: odom 좌표계로 변환됐으므로 frame ID도 odom
        pub.publish(msg);
    }

    // 메세지 & 시간 정보 기록
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        const TimePoint callback_start = SteadyClock::now();
        TimePoint stage_start = callback_start;
        TimePoint stage_end = callback_start;
        LatencyRecord latency;

        // 센서 timestamp와 실제 callback 처리 시각을 별도로 기록함
        ++callback_count_;
        latency.callback_index = callback_count_;
        latency.msg_stamp_sec = msg->header.stamp.toSec();
        latency.wall_stamp_sec = ros::WallTime::now().toSec();
        latency.raw_msg_points =
            static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);

        stage_start = SteadyClock::now();
        CloudXYZI::Ptr raw = preprocessRawCloud(msg); // raw cloud 전처리
        stage_end = SteadyClock::now();
        latency.preprocess_ms = elapsedMs(stage_start, stage_end);
        latency.filtered_points = raw->size();
        if (raw->size() < static_cast<size_t>(min_registration_points_))
        {
            ROS_WARN_STREAM_THROTTLE(
                1.0,
                "[scan_matching_accumulator] skipped sparse cloud. points="
                << raw->size());
            latency.callback_total_ms =
                elapsedMs(callback_start, SteadyClock::now());
            writeLatencyLog(latency);
            return;
        }

        // 현재 pose 추정
        const Eigen::Matrix4f T_odom_lidar =
            estimatePose(raw, msg->header.stamp, latency);

        // 현재 프레임을 버퍼에 저장
        stage_start = SteadyClock::now();
        Frame frame;
        frame.stamp = msg->header.stamp;
        frame.lidar_frame = msg->header.frame_id;
        frame.raw = raw;
        frame.T_odom_lidar = T_odom_lidar;
        // 현재 프레임의 정합이 끝난 뒤 버퍼에 추가됨
        // 이를 통해 현재 프레임은 다음 프레임이 들어왔을 때 registration target의 일부가 됨
        frames_.push_back(frame);

        // 오래된 프레임 제거
        const int max_frames = std::max(window_frames_, submap_frames_);
        while (static_cast<int>(frames_.size()) > max_frames)
            frames_.pop_front(); // 앞에서부터 제거
        
        latency.window_size = std::min<int>(window_frames_, frames_.size());
        stage_end = SteadyClock::now();
        latency.buffer_ms = elapsedMs(stage_start, stage_end);

        // pose 발행
        stage_start = SteadyClock::now();
        pose_pub_.publish(poseMsgFromMatrix(T_odom_lidar,
                                            msg->header.stamp,
                                            odom_frame_));
        stage_end = SteadyClock::now();
        latency.pose_publish_ms = elapsedMs(stage_start, stage_end);

        if (callback_count_ % publish_every_n_frames_ != 0)
        {
            latency.callback_total_ms =
                elapsedMs(callback_start, SteadyClock::now());
            writeLatencyLog(latency);
            return;
        }
        if (publish_only_full_window_ &&
            static_cast<int>(frames_.size()) < window_frames_)
        {
            latency.callback_total_ms =
                elapsedMs(callback_start, SteadyClock::now());
            writeLatencyLog(latency);
            return;
        }

        // Dense window 발행 -> 이후 online_static_feature_node.cpp의 입력으로 들어감
        stage_start = SteadyClock::now();
        CloudXYZI::Ptr dense = buildDenseWindow();
        stage_end = SteadyClock::now();
        latency.dense_window_build_ms = elapsedMs(stage_start, stage_end);
        latency.dense_points = dense->size();

        stage_start = SteadyClock::now();
        publishCloud(dense, dense_pub_, msg->header.stamp, odom_frame_);
        stage_end = SteadyClock::now();
        latency.dense_window_publish_ms = elapsedMs(stage_start, stage_end);
        latency.dense_published = 1;

        // Registration submap 발행
        if (publish_submap_)
        {
            stage_start = SteadyClock::now();
            CloudXYZ::Ptr submap_xyz = buildRegistrationSubmap();
            CloudXYZI::Ptr submap_xyzi(new CloudXYZI());
            submap_xyzi->reserve(submap_xyz->size());
            for (const auto &p : submap_xyz->points)
            {
                PointXYZI q;
                q.x = p.x;
                q.y = p.y;
                q.z = p.z;
                q.intensity = 0.0f;
                submap_xyzi->push_back(q);
            }
            submap_xyzi->width = static_cast<uint32_t>(submap_xyzi->size());
            submap_xyzi->height = 1;
            submap_xyzi->is_dense = true;
            publishCloud(submap_xyzi, submap_pub_, msg->header.stamp,
                         odom_frame_);
            stage_end = SteadyClock::now();
            latency.submap_publish_ms = elapsedMs(stage_start, stage_end);
        }

        latency.callback_total_ms =
            elapsedMs(callback_start, SteadyClock::now());
        writeLatencyLog(latency);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scan_matching_accumulator_node");
    ScanMatchingAccumulatorNode node;
    ros::spin();
    return 0;
}
