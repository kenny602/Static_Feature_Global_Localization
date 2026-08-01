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

#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace
{

using CloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using SteadyClock = std::chrono::steady_clock;

double elapsedMs(const SteadyClock::time_point &a, const SteadyClock::time_point &b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

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

void rpyFromIsometry(const Eigen::Isometry3d &T,
                     double &roll, double &pitch, double &yaw)
{
    const Eigen::Vector3d e = T.linear().eulerAngles(2, 1, 0); // yaw pitch roll
    yaw = e[0];
    pitch = e[1];
    roll = e[2];
}

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
        loadMap();
        initMatcher();
        initLog();

        map_pose_pub_ = pnh_.advertise<geometry_msgs::PoseStamped>("map_pose", 10);
        map_odom_pub_ = pnh_.advertise<geometry_msgs::PoseStamped>("map_odom_pose", 10);
        fitness_pub_ = pnh_.advertise<std_msgs::Float32>("fitness", 10);
        aligned_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>("aligned_source", 1);
        path_pub_ = pnh_.advertise<nav_msgs::Path>("path", 1);
        marker_pub_ = pnh_.advertise<visualization_msgs::MarkerArray>("markers", 1);

        odom_sub_ = nh_.subscribe(odom_pose_topic_, 50,
                                  &MapLocalizerNode::odomPoseCallback, this);

        if (source_mode_ == "features")
        {
            plane_sub_.subscribe(nh_, plane_topic_, 2);
            corner_sub_.subscribe(nh_, corner_topic_, 2);
            feature_sync_ = std::make_unique<FeatureSync>(
                FeatureSyncPolicy(10), plane_sub_, corner_sub_);
            feature_sync_->setMaxIntervalDuration(ros::Duration(0.05));
            feature_sync_->registerCallback(
                boost::bind(&MapLocalizerNode::featureCallback, this, _1, _2));
        }
        else
        {
            accumulated_sub_ = nh_.subscribe(accumulated_topic_, 2,
                                             &MapLocalizerNode::accumulatedCallback, this);
        }

        ROS_INFO_STREAM("[map_localizer] matcher=" << matcher_type_
                        << ", source_mode=" << source_mode_
                        << ", map_points=" << map_cloud_->size()
                        << " (leaf=" << map_voxel_leaf_ << ")");
    }

    ~MapLocalizerNode()
    {
        saveTrajectory();
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
        pnh_.param("sg_max_iterations", sg_max_iterations_, 20);
        pnh_.param("sg_voxel_resolution", sg_voxel_resolution_, 1.0); // VGICP 전용

        // Correction gating (odometry 예측 대비 점프 제한)
        pnh_.param("max_correction_trans", max_correction_trans_, 2.0);
        pnh_.param("max_correction_rot_deg", max_correction_rot_deg_, 10.0);
        pnh_.param("min_source_points", min_source_points_, 50);

        pnh_.param("odom_stamp_tolerance", odom_stamp_tolerance_, 0.2);
        pnh_.param<std::string>("log_csv_path", log_csv_path_, "");
        // 종료 시 추정 궤적을 PCD로 저장 (x y z intensity=frame index).
        // 교수님 결과(map_trajectory.pcd)와 동일 포맷이라 바로 겹쳐 비교할 수 있다.
        pnh_.param<std::string>("traj_pcd_path", traj_pcd_path_, "");

        if (map_pcd_path_.empty())
            throw std::runtime_error("[map_localizer] map_pcd_path is required.");
        if (source_mode_ != "features" && source_mode_ != "accumulated")
            throw std::runtime_error("[map_localizer] source_mode must be "
                                     "'features' or 'accumulated'.");
    }

    // -------------------------------------------------------------------- map
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

        if (map_voxel_leaf_ > 0.0)
        {
            pcl::VoxelGrid<pcl::PointXYZ> vg;
            vg.setLeafSize(map_voxel_leaf_, map_voxel_leaf_, map_voxel_leaf_);
            vg.setInputCloud(map_cloud_);
            CloudXYZ::Ptr filtered(new CloudXYZ());
            vg.filter(*filtered);
            map_cloud_ = filtered;
        }
        if (map_cloud_->empty())
            throw std::runtime_error("[map_localizer] map cloud is empty.");
    }

    void initMatcher()
    {
        if (matcher_type_ == "ndt")
        {
            ndt_.setResolution(ndt_resolution_);
            ndt_.setStepSize(ndt_step_size_);
            ndt_.setTransformationEpsilon(ndt_trans_eps_);
            ndt_.setMaximumIterations(ndt_max_iterations_);
            ndt_.setInputTarget(map_cloud_);
            return;
        }

        // small_gicp: 맵은 한 번만 전처리해서 캐시
        sg_setting_.type = parseSmallGicpType(matcher_type_);
        sg_setting_.downsampling_resolution = sg_downsample_resolution_;
        sg_setting_.max_correspondence_distance = sg_max_corr_dist_;
        sg_setting_.num_threads = sg_num_threads_;
        sg_setting_.max_iterations = sg_max_iterations_;
        sg_setting_.voxel_resolution = sg_voxel_resolution_;

        const auto map_vec = cloudToVec4f(*map_cloud_);
        std::tie(sg_target_, sg_target_tree_) =
            small_gicp::preprocess_points(map_vec, sg_downsample_resolution_,
                                          10, sg_num_threads_);

        // VGICP: target을 GaussianVoxelMap(복셀별 평균/공분산 해시맵)으로 미리 만들어
        // 캐시한다. correspondence를 KD-tree NN이 아니라 voxel 해시 조회로 처리한다.
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

    void initLog()
    {
        if (log_csv_path_.empty())
            return;
        log_.open(log_csv_path_, std::ios::out | std::ios::trunc);
        log_ << "frame,stamp,matcher,source_mode,source_points,"
             << "x,y,z,roll,pitch,yaw,"
             << "corr_dx,corr_dy,corr_dz,corr_dyaw,"
             << "fitness,converged,rejected,align_ms\n";
    }

    // ------------------------------------------------------------- callbacks
    void odomPoseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lk(odom_mutex_);
        odom_buffer_.emplace_back(msg->header.stamp, poseMsgToIsometry(msg->pose));
        while (odom_buffer_.size() > 500)
            odom_buffer_.pop_front();
    }

    bool lookupOdomPose(const ros::Time &stamp, Eigen::Isometry3d &T_odom_lidar)
    {
        std::lock_guard<std::mutex> lk(odom_mutex_);
        double best_dt = odom_stamp_tolerance_;
        bool found = false;
        for (const auto &entry : odom_buffer_)
        {
            const double dt = std::fabs((entry.first - stamp).toSec());
            if (dt <= best_dt)
            {
                best_dt = dt;
                T_odom_lidar = entry.second;
                found = true;
            }
        }
        return found;
    }

    void featureCallback(const sensor_msgs::PointCloud2::ConstPtr &plane_msg,
                         const sensor_msgs::PointCloud2::ConstPtr &corner_msg)
    {
        CloudXYZ::Ptr source(new CloudXYZ());
        CloudXYZ tmp;
        pcl::fromROSMsg(*plane_msg, tmp);
        *source += tmp;
        tmp.clear();
        pcl::fromROSMsg(*corner_msg, tmp);
        *source += tmp;
        process(source, plane_msg->header.stamp);
    }

    void accumulatedCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        CloudXYZ::Ptr source(new CloudXYZ());
        pcl::fromROSMsg(*msg, *source);
        process(source, msg->header.stamp);
    }

    // ------------------------------------------------------------------ core
    void process(CloudXYZ::Ptr source, const ros::Time &stamp)
    {
        if (static_cast<int>(source->size()) < min_source_points_)
        {
            ROS_WARN_THROTTLE(5.0, "[map_localizer] source too small (%zu pts)",
                              source->size());
            return;
        }

        if (source_voxel_leaf_ > 0.0)
        {
            pcl::VoxelGrid<pcl::PointXYZ> vg;
            vg.setLeafSize(source_voxel_leaf_, source_voxel_leaf_, source_voxel_leaf_);
            vg.setInputCloud(source);
            CloudXYZ::Ptr filtered(new CloudXYZ());
            vg.filter(*filtered);
            source = filtered;
        }

        Eigen::Isometry3d T_odom_lidar = Eigen::Isometry3d::Identity();
        const bool has_odom = lookupOdomPose(stamp, T_odom_lidar);
        if (!has_odom)
            ROS_WARN_THROTTLE(5.0, "[map_localizer] no odom pose near stamp; "
                                   "using last known T_odom_lidar.");
        if (has_odom)
            last_T_odom_lidar_ = T_odom_lidar;
        else
            T_odom_lidar = last_T_odom_lidar_;

        // 초기화: T_map_odom = T_map_lidar(init) * T_odom_lidar^-1
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

        const Eigen::Isometry3d prediction = T_map_odom_;

        double fitness = -1.0;
        bool converged = false;
        Eigen::Isometry3d T_map_odom_new = prediction;

        const auto t0 = SteadyClock::now();
        if (matcher_type_ == "ndt")
            converged = alignNdt(source, prediction, T_map_odom_new, fitness);
        else
            converged = alignSmallGicp(source, prediction, T_map_odom_new, fitness);
        const double align_ms = elapsedMs(t0, SteadyClock::now());

        // Gating: odometry 예측 대비 보정량 제한
        const Eigen::Isometry3d delta = prediction.inverse() * T_map_odom_new;
        const double corr_trans = delta.translation().norm();
        const double corr_rot_deg =
            Eigen::AngleAxisd(delta.linear()).angle() * 180.0 / M_PI;
        const bool rejected = (!converged) ||
                              corr_trans > max_correction_trans_ ||
                              corr_rot_deg > max_correction_rot_deg_;
        if (!rejected)
            T_map_odom_ = T_map_odom_new;
        else
            ROS_WARN_THROTTLE(2.0,
                "[map_localizer] correction rejected (conv=%d, dt=%.2fm, dr=%.1fdeg)",
                converged, corr_trans, corr_rot_deg);

        publishAndLog(source, stamp, T_odom_lidar, prediction,
                      fitness, converged, rejected, align_ms);
    }

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

    bool alignSmallGicp(const CloudXYZ::Ptr &source,
                        const Eigen::Isometry3d &guess,
                        Eigen::Isometry3d &result, double &fitness)
    {
        const auto source_vec = cloudToVec4f(*source);
        auto [sg_source, sg_source_tree] =
            small_gicp::preprocess_points(source_vec, sg_downsample_resolution_,
                                          10, sg_num_threads_);
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

    void publishAndLog(const CloudXYZ::Ptr &source, const ros::Time &stamp,
                       const Eigen::Isometry3d &T_odom_lidar,
                       const Eigen::Isometry3d &prediction,
                       double fitness, bool converged, bool rejected,
                       double align_ms)
    {
        frame_count_++;
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

        if (publish_tf_)
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
             << source->size() << ','
             << T_map_lidar.translation().x() << ','
             << T_map_lidar.translation().y() << ','
             << T_map_lidar.translation().z() << ','
             << roll << ',' << pitch << ',' << yaw << ','
             << delta.translation().x() << ','
             << delta.translation().y() << ','
             << delta.translation().z() << ','
             << dyaw << ','
             << fitness << ',' << (converged ? 1 : 0) << ','
             << (rejected ? 1 : 0) << ',' << align_ms << '\n';
        log_.flush();
    }

    // ---------------------------------------------------------------- members
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    using FeatureSyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2, sensor_msgs::PointCloud2>;
    using FeatureSync = message_filters::Synchronizer<FeatureSyncPolicy>;

    message_filters::Subscriber<sensor_msgs::PointCloud2> plane_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> corner_sub_;
    std::unique_ptr<FeatureSync> feature_sync_;
    ros::Subscriber accumulated_sub_;
    ros::Subscriber odom_sub_;

    ros::Publisher map_pose_pub_;
    ros::Publisher map_odom_pub_;
    ros::Publisher fitness_pub_;
    ros::Publisher aligned_pub_;
    ros::Publisher path_pub_;
    ros::Publisher marker_pub_;
    nav_msgs::Path path_;
    int max_path_poses_ = 0;  // 0 = 무제한

    std::string map_pcd_path_, map_pcd_path2_;
    std::string matcher_type_, source_mode_;
    std::string plane_topic_, corner_topic_, accumulated_topic_, odom_pose_topic_;
    std::string map_frame_id_;
    std::string odom_frame_id_;
    std::string log_csv_path_;

    bool publish_tf_ = true;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    double init_x_ = 0, init_y_ = 0, init_z_ = 0;
    double init_roll_ = 0, init_pitch_ = 0, init_yaw_ = 0;

    double map_voxel_leaf_ = 0.4;
    double source_voxel_leaf_ = 0.0;
    double ndt_resolution_ = 1.0, ndt_step_size_ = 0.1, ndt_trans_eps_ = 0.01;
    int ndt_max_iterations_ = 50;
    double sg_downsample_resolution_ = 0.25, sg_max_corr_dist_ = 1.0;
    int sg_num_threads_ = 4, sg_max_iterations_ = 20;
    double sg_voxel_resolution_ = 1.0;
    double max_correction_trans_ = 2.0, max_correction_rot_deg_ = 10.0;
    int min_source_points_ = 50;
    double odom_stamp_tolerance_ = 0.2;

    CloudXYZ::Ptr map_cloud_;
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_;
    small_gicp::RegistrationSetting sg_setting_;
    small_gicp::PointCloud::Ptr sg_target_;
    std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> sg_target_tree_;
    small_gicp::GaussianVoxelMap::Ptr sg_target_voxelmap_;  // VGICP 전용 캐시

    std::mutex odom_mutex_;
    std::deque<std::pair<ros::Time, Eigen::Isometry3d>> odom_buffer_;
    Eigen::Isometry3d last_T_odom_lidar_ = Eigen::Isometry3d::Identity();

    Eigen::Isometry3d T_map_odom_ = Eigen::Isometry3d::Identity();
    bool initialized_ = false;

    std::ofstream log_;
    long frame_count_ = 0;

    std::string traj_pcd_path_;
    pcl::PointCloud<pcl::PointXYZI> traj_cloud_;
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
