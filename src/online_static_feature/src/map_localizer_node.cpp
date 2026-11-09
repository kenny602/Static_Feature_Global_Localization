// map_localizer_node.cpp
//
// Scan-to-map localization on top of the online static feature pipeline.
//
// Frame convention:
//   - scan_matching_accumulator_node publishes T_odom_lidar (odometry).
//   - online_static_feature_node publishes feature clouds already expressed
//     in the odom frame (transform_with_pose=true).
//   - This node estimates the drift correction T_map_odom only:
//         T_map_lidar = T_map_odom * T_odom_lidar
//     Each incoming source cloud (odom frame) is registered against the
//     offline map with initial guess = previous T_map_odom.
//
// Matcher backends (param `matcher_type`):
//   "ndt"                       -> pcl::NormalDistributionsTransform
//   "gicp" | "plane_icp" |
//   "icp"  | "vgicp"            -> small_gicp (map preprocessed once)
//
// Source selection (param `source_mode`):
//   "features"    -> scan_plane_features + scan_corner_features (concatenated)
//   "static"      -> scan_static_features (분류 없이 동적 제거만 거친 단일 클라우드,
//                    online_static_feature_node의 use_feature_classification=0)
//   "accumulated" -> raw dense window cloud (baseline / ablation)
//
// Outputs:
//   ~map_pose        geometry_msgs/PoseStamped  (T_map_lidar)
//   ~map_odom_pose   geometry_msgs/PoseStamped  (T_map_odom, the correction)
//   ~fitness         std_msgs/Float32
//   ~aligned_source  sensor_msgs/PointCloud2    (debug, map frame)
//   CSV log per frame for the precision-evaluation harness.

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Geometry>

#include <cerrno>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace
{

using CloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using SteadyClock = std::chrono::steady_clock;

double elapsedMs(const SteadyClock::time_point &a, const SteadyClock::time_point &b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ROS의 geometry_msgs::Pose를 Eigen의 Isometry3d로 변환
Eigen::Isometry3d poseMsgToIsometry(const geometry_msgs::Pose &p)
{
    Eigen::Quaterniond q(p.orientation.w, p.orientation.x,
                         p.orientation.y, p.orientation.z);
    if (q.norm() < 1e-9)
        q = Eigen::Quaterniond::Identity();
    else
        q.normalize();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = q.toRotationMatrix();
    T.translation() = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
    return T;
}

// Eigen::Isometry3d를 ROS의 PoseStamped로 변환
// 최종 pose와 map-to-odom 보정 pose를 발행할 때 사용
geometry_msgs::PoseStamped isometryToPoseMsg(const Eigen::Isometry3d &T,
                                             const ros::Time &stamp,
                                             const std::string &frame_id)
{
    geometry_msgs::PoseStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id;
    msg.pose.position.x = T.translation().x();
    msg.pose.position.y = T.translation().y();
    msg.pose.position.z = T.translation().z();
    Eigen::Quaterniond q(T.linear());
    q.normalize();
    msg.pose.orientation.w = q.w();
    msg.pose.orientation.x = q.x();
    msg.pose.orientation.y = q.y();
    msg.pose.orientation.z = q.z();
    return msg;
}

// 회전행렬에서 roll, pitch, yaw를 추출
// CSV 로그에 자세를 기록하기 위함
void rpyFromIsometry(const Eigen::Isometry3d &T,
                     double &roll, double &pitch, double &yaw)
{
    // Eigen::eulerAngles(2,1,0) 은 pitch 를 [0,pi] 로 강제하는 탓에
    // 회전이 없어도 (pi, -pi, pi) 같은 뒤집힌 해를 돌려준다.
    // 실제로 단위행렬이 yaw=3.14 로 찍혀 "180도 돌아갔다"는 오독을 유발했다.
    // 직접 분해해서 yaw, roll 을 [-pi,pi], pitch 를 [-pi/2,pi/2] 로 둔다.
    const Eigen::Matrix3d R = T.linear();
    pitch = std::asin(std::max(-1.0, std::min(1.0, -R(2, 0))));
    if (std::abs(R(2, 0)) < 0.999999)   // 짐벌락이 아니면 일반 해
    {
        yaw = std::atan2(R(1, 0), R(0, 0));
        roll = std::atan2(R(2, 1), R(2, 2));
    }
    else                                 // pitch = +-90도: yaw 와 roll 이 축퇴
    {
        yaw = std::atan2(-R(0, 1), R(1, 1));
        roll = 0.0;
    }
}

// PCL cloud인 PointXYZ(x, y, z)를 small_gicp의 입력 형식 Eigen::Vector4f인 (x, y, z, 1)으로 변환함
std::vector<Eigen::Vector4f> cloudToVec4f(const CloudXYZ &cloud)
{
    std::vector<Eigen::Vector4f> out;
    out.reserve(cloud.size());
    for (const auto &p : cloud.points)
    {
        if (!pcl::isFinite(p))
            continue;
        out.emplace_back(p.x, p.y, p.z, 1.0f);
    }
    return out;
}

} // namespace

class MapLocalizerNode
{
public:
    MapLocalizerNode() : pnh_("~")
    {
        loadParams();
        // 맵 로드 + 다운샘플 + target 자료구조 구축은 시작 시 1회지만, 조건마다
        // 맵 크기가 다르므로(feature map vs raw map) 비용과 점 수를 기록해 둔다.
        const auto t_map0 = SteadyClock::now();
        loadMap();         // offline PCD map 로드
        initMatcher();     // matcher target 전처리
        map_prep_ms_ = elapsedMs(t_map0, SteadyClock::now());   // 로그 기록용
        map_points_ = map_cloud_->size();                       // 정합 target map
        initLog();         // 로깅 시작

        map_pose_pub_ = pnh_.advertise<geometry_msgs::PoseStamped>("map_pose", 10);         // 맵 매칭으로 구하는 최종 pose
        map_odom_pub_ = pnh_.advertise<geometry_msgs::PoseStamped>("map_odom_pose", 10);    // 스캔 매칭으로 넘겨받는 pose (odometry)
        fitness_pub_ = pnh_.advertise<std_msgs::Float32>("fitness", 10);
        aligned_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>("aligned_source", 1);
        path_pub_ = pnh_.advertise<nav_msgs::Path>("path", 1);
        marker_pub_ = pnh_.advertise<visualization_msgs::MarkerArray>("markers", 1);
        // latched: RViz를 나중에 켜도 받을 수 있다. 시작 시 1회 발행이라
        // 프레임별 처리시간에는 영향이 없다. map_voxel_leaf 적용 후의 실제
        // 정합 target이므로 원본 PCD를 따로 띄우는 것보다 디버깅에 정확하다.
        map_cloud_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>("map_cloud", 1, true);
        {
            const CloudXYZ::Ptr viz =
                (viz_voxel_leaf_ > 0.0 && viz_voxel_leaf_ > map_voxel_leaf_)
                    ? downsample(map_cloud_, viz_voxel_leaf_)
                    : map_cloud_;
            ROS_INFO_STREAM("[map_localizer] map_cloud 발행: " << viz->size()
                            << " points (viz_voxel_leaf=" << viz_voxel_leaf_ << ")");
            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(*viz, map_msg);
            map_msg.header.stamp = ros::Time::now();
            map_msg.header.frame_id = map_frame_id_;
            map_cloud_pub_.publish(map_msg);
        }

        // source cloud와 odom pose(T_odom_lidar)를 message_filters로 짝지어 받는다.
        // 예전에는 odom pose를 별도 구독해 버퍼에서 stamp로 찾고(±odom_stamp_tolerance),
        // 못 찾으면 직전 값을 재사용했다. 부하로 pose 전달이 늦으면 정합은 정상인데
        // 출력 T_map_lidar = T_map_odom * T_odom_lidar 가 그 자리에 고정되는 문제가 있었다
        // (ROS2/Isaac attempt_2: ~1 s 고정 후 1.8 m 점프). ROS2 포팅본(2026-11-06)과 같은
        // 구조로 맞춘다: 둘 다 도착한 프레임만 처리하고, 짝이 없으면 프레임을 누락시킨다.
        odom_mf_sub_.subscribe(nh_, odom_pose_topic_, 50);

        if (source_mode_ == "features")
        {
            plane_sub_.subscribe(nh_, plane_topic_, 5);
            corner_sub_.subscribe(nh_, corner_topic_, 5);
            feature_sync_ = std::make_unique<FeatureSync>(
                FeatureSyncPolicy(sync_queue_size_), plane_sub_, corner_sub_, odom_mf_sub_);
            feature_sync_->setMaxIntervalDuration(ros::Duration(odom_sync_slop_sec_));
            feature_sync_->registerCallback(
                boost::bind(&MapLocalizerNode::featureCallback, this, _1, _2, _3));
        }
        else
        {
            // static / accumulated: 단일 cloud 토픽 + odom pose
            const std::string &topic =
                (source_mode_ == "static") ? static_topic_ : accumulated_topic_;
            cloud_mf_sub_.subscribe(nh_, topic, 5);
            cloud_sync_ = std::make_unique<CloudSync>(
                CloudSyncPolicy(sync_queue_size_), cloud_mf_sub_, odom_mf_sub_);
            cloud_sync_->setMaxIntervalDuration(ros::Duration(odom_sync_slop_sec_));
            cloud_sync_->registerCallback(
                boost::bind(&MapLocalizerNode::cloudOdomCallback, this, _1, _2));
        }
        // 수신/처리 프레임 수 진단용 (종료 시 로그). 동기화와 무관한 단순 카운터.
        cloud_count_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(
            (source_mode_ == "features") ? plane_topic_
                : (source_mode_ == "static") ? static_topic_ : accumulated_topic_,
            5,
            [this](const sensor_msgs::PointCloud2::ConstPtr &) { ++clouds_received_; });

        ROS_INFO_STREAM("[map_localizer] matcher=" << matcher_type_
                        << ", source_mode=" << source_mode_
                        << ", map_points=" << map_cloud_->size()
                        << " (leaf=" << map_voxel_leaf_ << ")"
                        << ", sg_num_neighbors=" << sg_num_neighbors_);
    }

    ~MapLocalizerNode()
    {
        ROS_INFO("[map_localizer] source clouds received=%zu, processed (synced with odom)=%zu",
                 clouds_received_, clouds_processed_);
        writeRunSummary();
        saveTrajectory();
    }

    // 종료 시 수신/처리 프레임 수를 CSV 옆 map_localizer_summary.txt 에 남긴다.
    // ROS1에서는 destructor 시점의 ROS_INFO가 roslaunch 로그에 남지 않으므로(이미 shutdown 중)
    // 파일로 쓴다. received == processed 이면 동기화로 누락된 프레임이 없다는 뜻이다.
    void writeRunSummary() const
    {
        if (log_csv_path_.empty())
            return;
        const size_t slash = log_csv_path_.find_last_of('/');
        const std::string dir =
            (slash == std::string::npos) ? std::string(".") : log_csv_path_.substr(0, slash);
        std::ofstream f((dir + "/map_localizer_summary.txt").c_str(), std::ios::out | std::ios::trunc);
        if (!f.is_open())
            return;
        f << "source_clouds_received=" << clouds_received_ << '\n'
          << "processed_synced_with_odom=" << clouds_processed_ << '\n';
    }

private:
    // ------------------------------------------------------------------ params
    void loadParams()
    {
        pnh_.param<std::string>("map_pcd_path", map_pcd_path_, "");
        pnh_.param<std::string>("map_pcd_path2", map_pcd_path2_, "");
        pnh_.param<std::string>("matcher_type", matcher_type_, "ndt");
        pnh_.param<std::string>("source_mode", source_mode_, "features");

        pnh_.param<std::string>("plane_feature_topic", plane_topic_,
                                "/online_static_feature/scan_plane_features");
        pnh_.param<std::string>("corner_feature_topic", corner_topic_,
                                "/online_static_feature/scan_corner_features");
        pnh_.param<std::string>("static_feature_topic", static_topic_,
                                "/online_static_feature_node/scan_static_features");
        pnh_.param<std::string>("accumulated_topic", accumulated_topic_,
                                "/online_static_feature/accumulated_window");
        pnh_.param<std::string>("odom_pose_topic", odom_pose_topic_,
                                "/scan_matching_accumulator/pose");

        pnh_.param<std::string>("map_frame_id", map_frame_id_, "map");
        // map -> odom TF 발행. accumulator/feature 클라우드가 odom frame이므로
        // 이 TF가 있어야 RViz에서 map frame과 함께 볼 수 있다.
        pnh_.param<std::string>("odom_frame_id", odom_frame_id_, "odom");
        pnh_.param("publish_tf", publish_tf_, true);
        pnh_.param("max_path_poses", max_path_poses_, 0);  // 0 = 무제한

        // Initial T_map_lidar (LSMCL initialize를 대신하는 수동 초기값)
        pnh_.param("init_x", init_x_, 0.0);
        pnh_.param("init_y", init_y_, 0.0);
        pnh_.param("init_z", init_z_, 0.0);
        pnh_.param("init_roll", init_roll_, 0.0);
        pnh_.param("init_pitch", init_pitch_, 0.0);
        pnh_.param("init_yaw", init_yaw_, 0.0);

        pnh_.param("map_voxel_leaf", map_voxel_leaf_, 0.4);
        // RViz 표시용 map_cloud 토픽 전용 축약. 정합에는 쓰이지 않는다.
        // 정합용 0.25는 raw 맵에서 수천만 점이라 RViz가 못 버틴다.
        // 0 = 축약 없이 정합 target을 그대로 발행(기존 동작).
        pnh_.param("viz_voxel_leaf", viz_voxel_leaf_, 0.0);
        pnh_.param("source_voxel_leaf", source_voxel_leaf_, 0.0); // 0 = off

        // NDT
        pnh_.param("ndt_resolution", ndt_resolution_, 1.0);
        pnh_.param("ndt_step_size", ndt_step_size_, 0.1);
        pnh_.param("ndt_trans_eps", ndt_trans_eps_, 0.01);
        pnh_.param("ndt_max_iterations", ndt_max_iterations_, 50);

        // small_gicp
        // small_gicp repo 기본값 그대로 (RegistrationSetting 참조)
        pnh_.param("sg_downsample_resolution", sg_downsample_resolution_, 0.25);
        pnh_.param("sg_max_corr_dist", sg_max_corr_dist_, 1.0);
        pnh_.param("sg_num_threads", sg_num_threads_, 4);
        // 공분산 추정용 kNN 개수. preprocess_points의 라이브러리 기본값은 10,
        // small_gicp의 KITTI odometry 벤치마크는 20을 쓴다(benchmark_odom.hpp).
        // 고윳값은 (1e-3,1,1)로 정규화되어 버려지므로, 이 값이 정하는 건
        // 지역 평면 '법선 방향'을 얼마나 넓은 이웃으로 평균 내느냐다.
        pnh_.param("sg_num_neighbors", sg_num_neighbors_, 10);
        pnh_.param("sg_max_iterations", sg_max_iterations_, 20);
        pnh_.param("sg_voxel_resolution", sg_voxel_resolution_, 1.0); // VGICP 전용

        // Correction gating (odometry 예측 대비 점프 제한)
        pnh_.param("max_correction_trans", max_correction_trans_, 2.0);
        pnh_.param("max_correction_rot_deg", max_correction_rot_deg_, 10.0);
        pnh_.param("min_source_points", min_source_points_, 50);
        // true 면 small_gicp 가 수렴 판정을 못 준 프레임도 버린다(예전 동작).
        // false = small_gicp 원본에 가까운 동작.
        pnh_.param("reject_on_not_converged", reject_on_not_converged_, false);

        // cloud와 odom pose의 stamp 허용 차이. 둘 다 같은 입력 스캔의 stamp를 그대로 쓰므로
        // 원래 0이어야 하고, 0.02 s는 여유값이다. (odom_stamp_tolerance는 더 이상 쓰지 않는다)
        pnh_.param("odom_sync_slop_sec", odom_sync_slop_sec_, 0.02);
        pnh_.param("sync_queue_size", sync_queue_size_, 20);
        pnh_.param<std::string>("log_csv_path", log_csv_path_, "");
        // 종료 시 추정 궤적을 PCD로 저장 (x y z intensity=frame index).
        // 교수님 결과(map_trajectory.pcd)와 동일 포맷이라 바로 겹쳐 비교할 수 있다.
        pnh_.param<std::string>("traj_pcd_path", traj_pcd_path_, "");

        if (map_pcd_path_.empty())
            throw std::runtime_error("[map_localizer] map_pcd_path is required.");
        if (source_mode_ != "features" && source_mode_ != "static" &&
            source_mode_ != "accumulated")
            throw std::runtime_error("[map_localizer] source_mode must be "
                                     "'features', 'static' or 'accumulated'.");
    }

    // small_gicp 기반 복셀 축약. 반환 타입이 std::shared_ptr이라
    // pcl::PointCloud::Ptr(PCL 1.10은 boost)과 다를 수 있어 swap으로 옮긴다.
    CloudXYZ::Ptr downsample(const CloudXYZ::Ptr &in, double leaf) const
    {
        // small_gicp의 축약 함수는 PCL 1.11+ traits를 요구하거나(여긴 1.10)
        // 지역 람다 템플릿 문제로 GCC 9에서 안 붙어서 여기 직접 구현한다.
        // 방식은 small_gicp와 동일: 복셀 좌표를 축당 21bit로 64bit 키에 패킹한 뒤
        // 정렬해서 같은 키끼리 평균낸다. PCL VoxelGrid처럼 dx*dy*dz 선형 인덱스를
        // 만들지 않으므로 맵이 아무리 넓어도 int32 오버플로가 없다.
        // 유효 범위는 축당 +-2^20 복셀 = 0.25 m에서 +-262 km.
        constexpr std::int64_t kBits = 21;
        constexpr std::int64_t kMask = (std::int64_t{1} << kBits) - 1;
        constexpr std::int64_t kOff  = std::int64_t{1} << (kBits - 1);
        const double inv = 1.0 / leaf;

        std::vector<std::pair<std::uint64_t, std::uint32_t>> keys;
        keys.reserve(in->size());
        for (std::uint32_t i = 0; i < in->size(); ++i)
        {
            const auto &p = (*in)[i];
            if (!pcl::isFinite(p))
                continue;
            const std::int64_t cx = std::llround(std::floor(p.x * inv)) + kOff;
            const std::int64_t cy = std::llround(std::floor(p.y * inv)) + kOff;
            const std::int64_t cz = std::llround(std::floor(p.z * inv)) + kOff;
            if (cx < 0 || cx > kMask || cy < 0 || cy > kMask || cz < 0 || cz > kMask)
                continue;   // 범위 밖은 버린다(small_gicp와 동일 동작)
            keys.emplace_back(static_cast<std::uint64_t>(cx)
                              | (static_cast<std::uint64_t>(cy) << kBits)
                              | (static_cast<std::uint64_t>(cz) << (2 * kBits)), i);
        }

        std::sort(keys.begin(), keys.end(),
                  [](const std::pair<std::uint64_t, std::uint32_t> &a,
                     const std::pair<std::uint64_t, std::uint32_t> &b)
                  { return a.first < b.first; });

        CloudXYZ::Ptr out(new CloudXYZ());
        out->reserve(keys.size() / 4 + 1);
        for (size_t i = 0; i < keys.size();)
        {
            size_t j = i;
            double sx = 0, sy = 0, sz = 0;
            for (; j < keys.size() && keys[j].first == keys[i].first; ++j)
            {
                const auto &p = (*in)[keys[j].second];
                sx += p.x; sy += p.y; sz += p.z;
            }
            const double n = static_cast<double>(j - i);
            pcl::PointXYZ c;
            c.x = static_cast<float>(sx / n);
            c.y = static_cast<float>(sy / n);
            c.z = static_cast<float>(sz / n);
            out->push_back(c);
            i = j;
        }
        out->width = out->size();
        out->height = 1;
        out->is_dense = true;
        out->header = in->header;
        return out;
    }

    // offline map 불러오기
    void loadMap()
    {
        map_cloud_.reset(new CloudXYZ()); // 안전장치 1
        CloudXYZ tmp;
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_pcd_path_, tmp) < 0)
            throw std::runtime_error("[map_localizer] failed to load " + map_pcd_path_);
        *map_cloud_ += tmp;

        if (!map_pcd_path2_.empty())      // 안전장치 2
        {
            tmp.clear();
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_pcd_path2_, tmp) < 0)
                throw std::runtime_error("[map_localizer] failed to load " + map_pcd_path2_);
            *map_cloud_ += tmp;
        }

        // 다운샘플링. PCL VoxelGrid가 아니라 small_gicp를 쓴다.
        // PCL은 (span_x/leaf)*(span_y/leaf)*(span_z/leaf) 를 int32 선형 인덱스로
        // 만들기 때문에, 넓은 맵에서는 leaf가 작으면 곱이 2^31을 넘어
        // "Leaf size is too small" 경고만 내고 축약을 통째로 건너뛴다.
        // (raw LAS 맵 610x939x60 m @0.25 -> 2,198,928,000 > 2,147,483,647)
        // small_gicp는 좌표를 축당 21bit로 비트 패킹하므로 곱이 생기지 않고
        // 0.25 m에서 +-262 km까지 안전하다.
        if (map_voxel_leaf_ > 0.0)
            map_cloud_ = downsample(map_cloud_, map_voxel_leaf_);
        if (map_cloud_->empty())
            throw std::runtime_error("[map_localizer] map cloud is empty.");
    }

    // Matcher 초기화
    void initMatcher()
    {
        if (matcher_type_ == "ndt") // NDT 세팅
        {
            ndt_.setResolution(ndt_resolution_);
            ndt_.setStepSize(ndt_step_size_);
            ndt_.setTransformationEpsilon(ndt_trans_eps_);
            ndt_.setMaximumIterations(ndt_max_iterations_);
            ndt_.setInputTarget(map_cloud_);
            return;
        }

        // small_gicp 세팅: 맵은 한 번만 전처리해서 캐시
        sg_setting_.type = parseSmallGicpType(matcher_type_);
        sg_setting_.downsampling_resolution = sg_downsample_resolution_;
        sg_setting_.max_correspondence_distance = sg_max_corr_dist_;
        sg_setting_.num_threads = sg_num_threads_;
        sg_setting_.max_iterations = sg_max_iterations_;
        sg_setting_.voxel_resolution = sg_voxel_resolution_;

        const auto map_vec = cloudToVec4f(*map_cloud_);
        // target인 offline map을 다운샘플링
        std::tie(sg_target_, sg_target_tree_) =
            small_gicp::preprocess_points(map_vec, sg_downsample_resolution_,
                                          sg_num_neighbors_, sg_num_threads_);

        // Small-GICP 세팅
        // VGICP: target을 GaussianVoxelMap(복셀별 평균/공분산 해시맵)으로 미리 만들어 캐시해둠
        // 이때 correspondence를 KD-tree NN이 아니라 voxel 해시 조회로 처리한다.
        if (sg_setting_.type == small_gicp::RegistrationSetting::VGICP)
            sg_target_voxelmap_ =
                small_gicp::create_gaussian_voxelmap(*sg_target_, sg_voxel_resolution_);
    }

    static small_gicp::RegistrationSetting::RegistrationType
    parseSmallGicpType(const std::string &s)
    {
        if (s == "icp") return small_gicp::RegistrationSetting::ICP;
        if (s == "plane_icp") return small_gicp::RegistrationSetting::PLANE_ICP;
        if (s == "vgicp") return small_gicp::RegistrationSetting::VGICP;
        return small_gicp::RegistrationSetting::GICP;
    }

    // RViz용: 지나온 경로(Path) + 현재 pose 마커(화살표/텍스트).
    // 마커 색은 정합 상태를 나타낸다 — 초록=반영, 빨강=reject.
    void publishPathAndMarkers(const Eigen::Isometry3d &T_map_lidar,
                               const ros::Time &stamp, bool rejected)
    {
        const auto pose = isometryToPoseMsg(T_map_lidar, stamp, map_frame_id_);

        if (path_pub_.getNumSubscribers() > 0 || !path_.poses.empty())
        {
            path_.header.stamp = stamp;
            path_.header.frame_id = map_frame_id_;
            path_.poses.push_back(pose);
            if (max_path_poses_ > 0 &&
                static_cast<int>(path_.poses.size()) > max_path_poses_)
                path_.poses.erase(path_.poses.begin());
            path_pub_.publish(path_);
        }

        if (marker_pub_.getNumSubscribers() == 0)
            return;

        visualization_msgs::MarkerArray arr;

        visualization_msgs::Marker arrow;
        arrow.header.stamp = stamp;
        arrow.header.frame_id = map_frame_id_;
        arrow.ns = "map_localizer";
        arrow.id = 0;
        arrow.type = visualization_msgs::Marker::ARROW;
        arrow.action = visualization_msgs::Marker::ADD;
        arrow.pose = pose.pose;
        arrow.scale.x = 3.0;   // 길이
        arrow.scale.y = 0.5;   // 축 지름
        arrow.scale.z = 0.5;   // 머리 지름
        arrow.color.a = 1.0;
        arrow.color.r = rejected ? 1.0f : 0.0f;
        arrow.color.g = rejected ? 0.0f : 1.0f;
        arrow.color.b = 0.0f;
        arr.markers.push_back(arrow);

        visualization_msgs::Marker text;
        text.header = arrow.header;
        text.ns = "map_localizer";
        text.id = 1;
        text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text.action = visualization_msgs::Marker::ADD;
        text.pose = pose.pose;
        text.pose.position.z += 3.0;
        text.scale.z = 1.5;   // 글자 크기
        text.color.a = 1.0;
        text.color.r = text.color.g = text.color.b = 1.0;
        char buf[160];
        std::snprintf(buf, sizeof(buf), "#%ld %s/%s%s",
                      frame_count_, matcher_type_.c_str(), source_mode_.c_str(),
                      rejected ? "\nREJECTED" : "");
        text.text = buf;
        arr.markers.push_back(text);

        marker_pub_.publish(arr);
    }

    void saveTrajectory()
    {
        if (traj_pcd_path_.empty() || traj_cloud_.empty())
            return;
        if (!ensureParentDirectory(traj_pcd_path_))
            return;
        traj_cloud_.width = static_cast<uint32_t>(traj_cloud_.size());
        traj_cloud_.height = 1;
        traj_cloud_.is_dense = true;
        if (pcl::io::savePCDFileBinary(traj_pcd_path_, traj_cloud_) < 0)
            ROS_ERROR_STREAM("[map_localizer] failed to save trajectory: "
                             << traj_pcd_path_);
        else
            ROS_INFO_STREAM("[map_localizer] saved trajectory ("
                            << traj_cloud_.size() << " poses): " << traj_pcd_path_);
    }

    // 상위 디렉토리를 재귀 생성한다. accumulator/feature 노드와 동일한 동작.
    // 이게 없으면 output_dir이 없을 때 CSV가 조용히 안 써진다.
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
            const size_t next = dir.find('/', start);
            const std::string part = dir.substr(start, next - start);
            if (!part.empty())
            {
                if (current.size() > 1)
                    current += "/";
                current += part;
                if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
                {
                    ROS_ERROR_STREAM("[map_localizer] mkdir failed: " << current
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

    void initLog()
    {
        if (log_csv_path_.empty())
            return;
        if (!ensureParentDirectory(log_csv_path_))
            return;
        log_.open(log_csv_path_, std::ios::out | std::ios::trunc);
        if (!log_.is_open())
        {
            ROS_ERROR_STREAM("[map_localizer] failed to open log: " << log_csv_path_);
            return;
        }
        // 기본 ostream 정밀도는 유효숫자 6자리라 ROS 타임스탬프(~1.7e9)가
        // 1.69e+09로 뭉개진다. stamp는 다른 노드 CSV와의 조인 키이고 x/y/z는
        // 정밀도 평가 대상이므로 고정소수점 6자리로 고정한다.
        // accumulator/feature 노드도 같은 설정을 쓴다.
        log_ << std::fixed << std::setprecision(6);
        // map_points / map_prep_ms는 시작 시 1회 확정되는 값이라 매 행에 같은 값이
        // 반복된다. 조건별 맵 크기와 초기화 비용을 CSV만 보고 알 수 있게 남긴다.
        // preprocess_ms: small_gicp preprocess_points (복셀 축약 + KD-tree + 공분산)
        // align_ms     : 순수 정합 반복. NDT는 preprocess_ms=0.
        // src_points_used: 위 축약을 거쳐 실제로 정합에 들어간 점 수.
        log_ << "frame,stamp,matcher,source_mode,source_points,src_points_used,"
             << "map_points,map_prep_ms,"
             << "x,y,z,roll,pitch,yaw,rot_deg,"
             << "corr_dx,corr_dy,corr_dz,corr_dyaw,"
             << "fitness,converged,rejected,preprocess_ms,align_ms,total_ms\n";
        ROS_INFO_STREAM("[map_localizer] log: " << log_csv_path_);
    }

    // ------------------------------------------------------------- callbacks
    // plane과 corenr 메세지를 각각 PCL로 변환 후 하나로 합침
    // 현재 기하학적 특징 라벨을 사용하지 않기 때문
    // odom pose는 message_filters로 같은 stamp의 것이 함께 들어온다.
    void featureCallback(const sensor_msgs::PointCloud2::ConstPtr &plane_msg,
                         const sensor_msgs::PointCloud2::ConstPtr &corner_msg,
                         const geometry_msgs::PoseStamped::ConstPtr &odom_msg)
    {
        CloudXYZ::Ptr source(new CloudXYZ());
        CloudXYZ tmp;
        pcl::fromROSMsg(*plane_msg, tmp);
        *source += tmp;
        tmp.clear();
        pcl::fromROSMsg(*corner_msg, tmp);
        *source += tmp;
        process(source, plane_msg->header.stamp, poseMsgToIsometry(odom_msg->pose));
    }

    // static: online static feature node의 정적 point / accumulated: raw dense window
    // odom pose는 message_filters로 같은 stamp의 것이 함께 들어온다.
    void cloudOdomCallback(const sensor_msgs::PointCloud2::ConstPtr &msg,
                           const geometry_msgs::PoseStamped::ConstPtr &odom_msg)
    {
        CloudXYZ::Ptr source(new CloudXYZ());
        pcl::fromROSMsg(*msg, *source);
        process(source, msg->header.stamp, poseMsgToIsometry(odom_msg->pose));
    }

    // 핵심 처리 함수
    // 모든 source mode (static or accumulated)는 최종적으로 이 함수로 들어옴
    // T_odom_lidar: source cloud와 같은 stamp의 scan matching odometry (동기화되어 전달됨)
    void process(CloudXYZ::Ptr source, const ros::Time &stamp,
                 const Eigen::Isometry3d &T_odom_lidar)
    {
        ++clouds_processed_;
        if (static_cast<int>(source->size()) < min_source_points_)
        {
            ROS_WARN_THROTTLE(5.0, "[map_localizer] source too small (%zu pts)",
                              source->size());
            return;
        }

        // Source 다운샘플링
        if (source_voxel_leaf_ > 0.0)
        {
            pcl::VoxelGrid<pcl::PointXYZ> vg;
            vg.setLeafSize(source_voxel_leaf_, source_voxel_leaf_, source_voxel_leaf_);
            vg.setInputCloud(source);
            CloudXYZ::Ptr filtered(new CloudXYZ());
            vg.filter(*filtered);
            source = filtered;
        }

        // 최초 T_map_odom 초기화: 사용자가 지정한 초기 map pose를 만듦 
        // T_map_odom = T_map_lidar(init) * T_odom_lidar^-1
        if (!initialized_)
        {
            Eigen::Isometry3d T_map_lidar_init = Eigen::Isometry3d::Identity();
            T_map_lidar_init.translation() =
                Eigen::Vector3d(init_x_, init_y_, init_z_);
            T_map_lidar_init.linear() =
                (Eigen::AngleAxisd(init_yaw_, Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(init_pitch_, Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(init_roll_, Eigen::Vector3d::UnitX()))
                    .toRotationMatrix();
            T_map_odom_ = T_map_lidar_init * T_odom_lidar.inverse();
            initialized_ = true;
        }

        // Initial guess 설정
        // 매 프레임 직전에 성공했던 T_map_odom을 다음 map matching의 초기값으로 사용
        const Eigen::Isometry3d prediction = T_map_odom_;

        double fitness = -1.0;
        bool converged = false;
        Eigen::Isometry3d T_map_odom_new = prediction;

        // preprocess와 align을 분리해서 잰다. small_gicp 경로는 매 프레임
        // source에 대해 복셀 축약 + KD-tree + 공분산을 새로 만드는데, 점 수가
        // 많을수록 이쪽이 정합 반복보다 더 커질 수 있다. 합쳐서 재면 raw가
        // 느린 이유를 "점이 많아서"인지 "전처리 때문"인지 귀속할 수 없다.
        double preprocess_ms = 0.0;
        size_t src_points_used = source->size();

        const auto t0 = SteadyClock::now();
        // Map matching 실행
        if (matcher_type_ == "ndt")
            converged = alignNdt(source, prediction, T_map_odom_new, fitness);
        else
            converged = alignSmallGicp(source, prediction, T_map_odom_new, fitness,
                                       preprocess_ms, src_points_used);
        const double total_ms = elapsedMs(t0, SteadyClock::now());
        const double align_ms = total_ms - preprocess_ms;

        // Gating: odometry 예측 대비 보정량 제한 - map matching 오차로 인해 확 튀어서 발산하는 것 방지
        const Eigen::Isometry3d delta = prediction.inverse() * T_map_odom_new;
        const double corr_trans = delta.translation().norm();
        const double corr_rot_deg =
            Eigen::AngleAxisd(delta.linear()).angle() * 180.0 / M_PI;
        // small_gicp 의 converged 는 "마지막 갱신량이 eps 아래였나"일 뿐이라
        // 정합 품질과 무관하다. 원저자의 KITTI odometry 벤치마크도 이 값을 보지 않고
        // result.T_target_source 를 그대로 누적한다. 그래서 기본값은 무시(false)다.
        // 게이팅(맵의 엉뚱한 곳으로 튀는 것 차단)만 남긴다.
        const bool rejected = (reject_on_not_converged_ && !converged) ||
                              corr_trans > max_correction_trans_ ||
                              corr_rot_deg > max_correction_rot_deg_;
        if (!rejected)
            T_map_odom_ = T_map_odom_new;
        else
            ROS_WARN_THROTTLE(2.0,
                "[map_localizer] correction rejected (conv=%d, dt=%.2fm, dr=%.1fdeg)",
                converged, corr_trans, corr_rot_deg);

        publishAndLog(source, stamp, T_odom_lidar, prediction,
                      fitness, converged, rejected,
                      preprocess_ms, align_ms, total_ms, src_points_used);
    }

    // NDT일때 map matching 함수
    bool alignNdt(const CloudXYZ::Ptr &source,
                  const Eigen::Isometry3d &guess,
                  Eigen::Isometry3d &result, double &fitness)
    {
        ndt_.setInputSource(source);
        CloudXYZ aligned;
        ndt_.align(aligned, guess.matrix().cast<float>());
        result.matrix() = ndt_.getFinalTransformation().cast<double>();
        fitness = ndt_.getFitnessScore();
        return ndt_.hasConverged();
    }

    // Small-GICP일때 map matching 함수
    bool alignSmallGicp(const CloudXYZ::Ptr &source,
                        const Eigen::Isometry3d &guess,
                        Eigen::Isometry3d &result, double &fitness,
                        double &preprocess_ms, size_t &src_points_used)
    {
        const auto t_pre = SteadyClock::now();
        const auto source_vec = cloudToVec4f(*source);
        auto [sg_source, sg_source_tree] =
            small_gicp::preprocess_points(source_vec, sg_downsample_resolution_,
                                          sg_num_neighbors_, sg_num_threads_);
        preprocess_ms = elapsedMs(t_pre, SteadyClock::now());
        src_points_used = sg_source->size();
        (void)sg_source_tree;
        // VGICP는 GaussianVoxelMap 전용 오버로드를, 나머지(GICP/ICP/PLANE_ICP)는
        // target KD-tree 오버로드를 호출해야 한다. tree 오버로드에 VGICP를 넘기면
        // small_gicp가 identity를 반환하므로 반드시 분기한다.
        small_gicp::RegistrationResult res;
        if (sg_setting_.type == small_gicp::RegistrationSetting::VGICP)
            res = small_gicp::align(*sg_target_voxelmap_, *sg_source,
                                    guess, sg_setting_);
        else
            res = small_gicp::align(*sg_target_, *sg_source, *sg_target_tree_,
                                    guess, sg_setting_);
        result = res.T_target_source;
        // small_gicp fitness 대용: 평균 residual error
        fitness = (res.num_inliers > 0)
                      ? res.error / static_cast<double>(res.num_inliers)
                      : std::numeric_limits<double>::max();
        return res.converged;
    }

    // 최종 pose 계산 & 출력
    void publishAndLog(const CloudXYZ::Ptr &source, const ros::Time &stamp,
                       const Eigen::Isometry3d &T_odom_lidar,
                       const Eigen::Isometry3d &prediction,
                       double fitness, bool converged, bool rejected,
                       double preprocess_ms, double align_ms, double total_ms,
                       size_t src_points_used)
    {
        frame_count_++;
        // 스캔 매칭으로 구한 오도메트리 pose에 맵 매칭으로 구한 최종 변환을 곱해서 map pose를 구함
        const Eigen::Isometry3d T_map_lidar = T_map_odom_ * T_odom_lidar;

        // 궤적 누적: intensity에 frame index를 담아 시간 순서를 남긴다.
        if (!traj_pcd_path_.empty())
        {
            pcl::PointXYZI tp;
            tp.x = static_cast<float>(T_map_lidar.translation().x());
            tp.y = static_cast<float>(T_map_lidar.translation().y());
            tp.z = static_cast<float>(T_map_lidar.translation().z());
            tp.intensity = static_cast<float>(traj_cloud_.size());
            traj_cloud_.push_back(tp);
        }

        map_pose_pub_.publish(isometryToPoseMsg(T_map_lidar, stamp, map_frame_id_));
        map_odom_pub_.publish(isometryToPoseMsg(T_map_odom_, stamp, map_frame_id_));

        // 오도메트리 좌표계의 accumulator / feature cloud를 Rviz의 맵 프레임에서 볼 수 있음
        if (publish_tf_) // map -> odom TF 발행
        {
            geometry_msgs::TransformStamped tf_msg;
            tf_msg.header.stamp = stamp;
            tf_msg.header.frame_id = map_frame_id_;
            tf_msg.child_frame_id = odom_frame_id_;
            tf_msg.transform.translation.x = T_map_odom_.translation().x();
            tf_msg.transform.translation.y = T_map_odom_.translation().y();
            tf_msg.transform.translation.z = T_map_odom_.translation().z();
            Eigen::Quaterniond q(T_map_odom_.linear());
            q.normalize();
            tf_msg.transform.rotation.w = q.w();
            tf_msg.transform.rotation.x = q.x();
            tf_msg.transform.rotation.y = q.y();
            tf_msg.transform.rotation.z = q.z();
            tf_broadcaster_.sendTransform(tf_msg);
        }

        std_msgs::Float32 f;
        f.data = static_cast<float>(fitness);
        fitness_pub_.publish(f);

        publishPathAndMarkers(T_map_lidar, stamp, rejected);

        // 입력 source에 T_map_odom을 적용하여 map 좌표계로 옮긴 디버깅용 cloud
        // 이 cloud가 map_cloud와 잘 겹치는지 확인하여 map matching을 시각적으로 체크할 수 있음
        if (aligned_pub_.getNumSubscribers() > 0)
        {
            CloudXYZ aligned;
            pcl::transformPointCloud(*source, aligned,
                                     T_map_odom_.matrix().cast<float>());
            sensor_msgs::PointCloud2 msg;
            pcl::toROSMsg(aligned, msg);
            msg.header.stamp = stamp;
            msg.header.frame_id = map_frame_id_;
            aligned_pub_.publish(msg);
        }

        if (!log_.is_open())
            return;
        double roll, pitch, yaw;
        rpyFromIsometry(T_map_lidar, roll, pitch, yaw);
        const Eigen::Isometry3d delta = prediction.inverse() * T_map_odom_;
        double droll, dpitch, dyaw;
        rpyFromIsometry(delta, droll, dpitch, dyaw);
        log_ << frame_count_ << ',' << stamp.toSec() << ','
             << matcher_type_ << ',' << source_mode_ << ','
             << source->size() << ',' << src_points_used << ','
             << map_points_ << ',' << map_prep_ms_ << ','
             << T_map_lidar.translation().x() << ','
             << T_map_lidar.translation().y() << ','
             << T_map_lidar.translation().z() << ','
             << roll << ',' << pitch << ',' << yaw << ','
             // 오일러 표기와 무관한 절대 회전량. 이 값이 진짜 자세 크기다.
             << Eigen::AngleAxisd(T_map_lidar.linear()).angle() * 180.0 / M_PI << ','
             << delta.translation().x() << ','
             << delta.translation().y() << ','
             << delta.translation().z() << ','
             << dyaw << ','
             << fitness << ',' << (converged ? 1 : 0) << ','
             << (rejected ? 1 : 0) << ','
             << preprocess_ms << ',' << align_ms << ',' << total_ms << '\n';
        log_.flush();
    }

    // ---------------------------------------------------------------- members
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    // features: plane + corner + odom pose
    using FeatureSyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, geometry_msgs::PoseStamped>;
    using FeatureSync = message_filters::Synchronizer<FeatureSyncPolicy>;
    // static / accumulated: cloud + odom pose
    using CloudSyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2, geometry_msgs::PoseStamped>;
    using CloudSync = message_filters::Synchronizer<CloudSyncPolicy>;

    message_filters::Subscriber<sensor_msgs::PointCloud2> plane_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> corner_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_mf_sub_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> odom_mf_sub_;
    std::unique_ptr<FeatureSync> feature_sync_;
    std::unique_ptr<CloudSync> cloud_sync_;
    ros::Subscriber cloud_count_sub_;
    size_t clouds_received_ = 0;
    size_t clouds_processed_ = 0;

    ros::Publisher map_pose_pub_;
    ros::Publisher map_odom_pub_;
    ros::Publisher fitness_pub_;
    ros::Publisher aligned_pub_;
    ros::Publisher path_pub_;
    ros::Publisher marker_pub_;
    ros::Publisher map_cloud_pub_;
    nav_msgs::Path path_;
    int max_path_poses_ = 0;  // 0 = 무제한

    std::string map_pcd_path_, map_pcd_path2_;
    std::string matcher_type_, source_mode_;
    std::string plane_topic_, corner_topic_, static_topic_;
    std::string accumulated_topic_, odom_pose_topic_;
    std::string map_frame_id_;
    std::string odom_frame_id_;
    std::string log_csv_path_;

    bool publish_tf_ = true;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    // 화살표 마커를 진행 방향으로 그리기 위한 상태 (시각화 전용)

    double init_x_ = 0, init_y_ = 0, init_z_ = 0;
    double init_roll_ = 0, init_pitch_ = 0, init_yaw_ = 0;

    double map_voxel_leaf_ = 0.4;
    double viz_voxel_leaf_ = 0.0;
    double source_voxel_leaf_ = 0.0;
    double ndt_resolution_ = 1.0, ndt_step_size_ = 0.1, ndt_trans_eps_ = 0.01;
    int ndt_max_iterations_ = 50;
    double sg_downsample_resolution_ = 0.25, sg_max_corr_dist_ = 1.0;
    int sg_num_threads_ = 4, sg_max_iterations_ = 20;
    int sg_num_neighbors_ = 10;
    double sg_voxel_resolution_ = 1.0;
    double max_correction_trans_ = 2.0, max_correction_rot_deg_ = 10.0;
    bool reject_on_not_converged_ = false;
    int min_source_points_ = 50;
    double odom_sync_slop_sec_ = 0.02;
    int sync_queue_size_ = 20;

    CloudXYZ::Ptr map_cloud_;
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_;
    small_gicp::RegistrationSetting sg_setting_;
    small_gicp::PointCloud::Ptr sg_target_;
    std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> sg_target_tree_;
    small_gicp::GaussianVoxelMap::Ptr sg_target_voxelmap_;  // VGICP 전용 캐시

    Eigen::Isometry3d T_map_odom_ = Eigen::Isometry3d::Identity();
    bool initialized_ = false;

    std::ofstream log_;
    long frame_count_ = 0;

    std::string traj_pcd_path_;
    pcl::PointCloud<pcl::PointXYZI> traj_cloud_;

    double map_prep_ms_ = 0.0;   // 맵 로드+다운샘플+target 구축 (시작 시 1회)
    size_t map_points_ = 0;      // map_voxel_leaf 적용 후 맵 점 수
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "map_localizer_node");
    try
    {
        MapLocalizerNode node;
        ros::spin();
    }
    catch (const std::exception &e)
    {
        ROS_FATAL_STREAM(e.what());
        return 1;
    }
    return 0;
}
