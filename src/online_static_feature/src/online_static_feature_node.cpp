#include "pillar_pca.hpp"
#include "types.hpp"
#include "voxel_map.hpp"

#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <sys/stat.h>
#include <sys/types.h>

class OnlineStaticFeatureNode
{
public:
    OnlineStaticFeatureNode()
        : nh_(),
          pnh_("~"),
          cloud_sub_(),
          pose_sub_(),
          sync_(SyncPolicy(20), cloud_sub_, pose_sub_)
    {
        loadRosParams();
        configureLocalOrigin();
        initLatencyLog();

        cloud_sub_.subscribe(nh_, cloud_topic_, 1);
        pose_sub_.subscribe(nh_, pose_topic_, 10);
        sync_.setMaxIntervalDuration(ros::Duration(sync_slop_sec_));
        sync_.registerCallback(boost::bind(
            &OnlineStaticFeatureNode::scanPoseCallback, this, _1, _2));

        accumulated_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            accumulated_topic_, 1, false);
        colored_feature_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            colored_feature_topic_, 1, false);
        plane_feature_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            plane_feature_topic_, 1, false);
        corner_feature_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            corner_feature_topic_, 1, false);
        feature_history_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            feature_history_topic_, 1, true);

        ROS_INFO_STREAM("[online_static_feature] cloud_topic=" << cloud_topic_
                        << ", pose_topic=" << pose_topic_
                        << ", window_frames=" << window_frames_
                        << ", transform_with_pose=" << transform_with_pose_
                        << ", local_origin_mode=" << local_origin_mode_);
    }

    ~OnlineStaticFeatureNode()
    {
        if (save_on_shutdown_pcd_)
            saveFeatureHistoryPcd("shutdown");
    }

private:
    using CloudMsg = sensor_msgs::PointCloud2;
    using PoseMsg = geometry_msgs::PoseStamped;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<CloudMsg, PoseMsg>;
    using SteadyClock = std::chrono::steady_clock;
    using TimePoint = SteadyClock::time_point;

    struct FeatureTiming
    {
        double build_voxel_pillar_ms = 0.0;
        double classify_pillars_ms = 0.0;
        double split_features_ms = 0.0;
        double publish_features_ms = 0.0;
        double total_ms = 0.0;

        size_t voxel_count = 0;
        size_t pillar_count = 0;
        size_t feature_points = 0;
        size_t plane_points = 0;
        size_t corner_points = 0;

        int planar_count = 0;
        int corner_count = 0;
        int voxel_min_removed = 0;
        int voxel_dynamic_removed = 0;
        int pillar_min_removed = 0;
        int pillar_dynamic_removed = 0;
        int span_removed = 0;
        int fill_removed = 0;
        int voxel_corner_points = 0;
        int wall_pillar_count = 0;
        int corner_ds_removed = 0;
    };

    struct LatencyRecord
    {
        int callback_index = 0;
        int window_index = 0;
        int window_size = 0;
        double msg_stamp_sec = 0.0;
        double log_stamp_sec = 0.0;

        size_t input_points = 0;
        size_t map_points = 0;
        size_t accumulated_points = 0;

        double ros_to_pcl_ms = 0.0;
        double transform_ms = 0.0;
        double buffer_ms = 0.0;
        double accumulate_ms = 0.0;
        double floor_align_ms = 0.0;
        double ceiling_crop_ms = 0.0;
        double downsample_ms = 0.0;
        double publish_accumulated_ms = 0.0;
        double window_algorithm_ms = 0.0;
        double window_total_ms = 0.0;

        FeatureTiming feature;
    };

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    std::string cloud_topic_ = "/Pointcloud2";
    std::string pose_topic_ = "/lidar_pose";
    std::string map_frame_ = "map";
    std::string accumulated_topic_ = "accumulated_cloud";
    std::string colored_feature_topic_ = "static_features_colored";
    std::string plane_feature_topic_ = "scan_plane_features";
    std::string corner_feature_topic_ = "scan_corner_features";
    std::string feature_history_topic_ = "feature_map_colored";
    std::string local_origin_mode_ = "none";
    std::string origin_las_path_;
    std::string save_feature_history_path_ =
        "/home/rain/online_static_feature_ws/output/online_static_features.pcd";
    std::string latency_log_path_ =
        "/home/rain/online_static_feature_ws/output/online_static_feature_latency.csv";

    int window_frames_ = 3;
    int publish_every_n_frames_ = 1;
    int callback_count_ = 0;
    int processed_window_count_ = 0;
    int processed_feature_frames_ = 0;
    int max_feature_history_points_ = 2000000;
    bool transform_with_pose_ = true;
    bool subtract_local_origin_ = false;
    bool publish_feature_history_ = true;
    bool save_on_shutdown_pcd_ = true;
    bool latency_log_enabled_ = true;
    double sync_slop_sec_ = 0.05;
    double local_origin_x_ = 0.0;
    double local_origin_y_ = 0.0;
    double local_origin_z_ = 0.0;
    Eigen::Vector3f local_origin_ = Eigen::Vector3f::Zero();

    ClassifyParams params_;

    message_filters::Subscriber<CloudMsg> cloud_sub_;
    message_filters::Subscriber<PoseMsg> pose_sub_;
    message_filters::Synchronizer<SyncPolicy> sync_;

    ros::Publisher accumulated_pub_;
    ros::Publisher colored_feature_pub_;
    ros::Publisher plane_feature_pub_;
    ros::Publisher corner_feature_pub_;
    ros::Publisher feature_history_pub_;

    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> frame_buffer_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr feature_history_{
        new pcl::PointCloud<pcl::PointXYZRGB>()};
    std::ofstream latency_log_;

    void loadRosParams()
    {
        pnh_.param<std::string>("cloud_topic", cloud_topic_, cloud_topic_);
        pnh_.param<std::string>("pose_topic", pose_topic_, pose_topic_);
        pnh_.param<std::string>("map_frame", map_frame_, map_frame_);
        pnh_.param<std::string>("accumulated_topic", accumulated_topic_, accumulated_topic_);
        pnh_.param<std::string>("colored_feature_topic", colored_feature_topic_, colored_feature_topic_);
        pnh_.param<std::string>("plane_feature_topic", plane_feature_topic_, plane_feature_topic_);
        pnh_.param<std::string>("corner_feature_topic", corner_feature_topic_, corner_feature_topic_);
        pnh_.param<std::string>("feature_history_topic", feature_history_topic_, feature_history_topic_);
        pnh_.param<std::string>("local_origin_mode", local_origin_mode_, local_origin_mode_);
        pnh_.param<std::string>("origin_las_path", origin_las_path_, origin_las_path_);
        pnh_.param<std::string>("save_feature_history_path", save_feature_history_path_,
                                save_feature_history_path_);
        pnh_.param<std::string>("latency_log_path", latency_log_path_, latency_log_path_);
        pnh_.param("window_frames", window_frames_, window_frames_);
        pnh_.param("publish_every_n_frames", publish_every_n_frames_, publish_every_n_frames_);
        pnh_.param("max_feature_history_points", max_feature_history_points_,
                   max_feature_history_points_);
        pnh_.param("transform_with_pose", transform_with_pose_, transform_with_pose_);
        pnh_.param("publish_feature_history", publish_feature_history_, publish_feature_history_);
        pnh_.param("save_on_shutdown_pcd", save_on_shutdown_pcd_,
                   save_on_shutdown_pcd_);
        pnh_.param("latency_log_enabled", latency_log_enabled_, latency_log_enabled_);
        pnh_.param("sync_slop_sec", sync_slop_sec_, sync_slop_sec_);
        pnh_.param("local_origin_x", local_origin_x_, local_origin_x_);
        pnh_.param("local_origin_y", local_origin_y_, local_origin_y_);
        pnh_.param("local_origin_z", local_origin_z_, local_origin_z_);

        window_frames_ = std::max(1, window_frames_);
        publish_every_n_frames_ = std::max(1, publish_every_n_frames_);

        // Defaults mirror the GIS tuning used in generalize_static_feature_ws,
        // except floor alignment is off because the pose already places scans in map.
        params_.use_downsample = 0;
        params_.use_floor_align = 0;
        params_.voxel_size = 0.2f;
        params_.n_voxel_min = 5;
        params_.n_pillar_min = 10;
        params_.use_voxel_filter = 0;
        params_.voxel_s_thresh = 0.25;
        params_.simulate_no_ceiling = 0;
        params_.ceiling_cut_percentile = 18.0f;
        params_.use_span_filter = 1;
        params_.H_min_voxel = 0;
        params_.fill_ratio_thresh = 0.4;
        params_.horiz_normal_thresh = 0.9;
        params_.use_floor_height_filter = 1;
        params_.floor_height_radius = 10;
        params_.floor_height_max = 0.7;
        params_.use_phi_weight = 0;
        params_.phi_mean_thresh = 0.4;
        params_.phi_std_thresh = 0.4;
        params_.s_thresh = 0.0005;
        params_.use_weight = 0;
        params_.n_ref = 20.0;
        params_.mu_thresh = 0.55;
        params_.sigma_thresh = 0.55;
        params_.use_voxel_corner = 1;
        params_.corner_multiplier = 5.0;
        params_.use_corner_downsample = 0;

        pnh_.param("use_downsample", params_.use_downsample, params_.use_downsample);
        pnh_.param("downsample_size", params_.downsample_size, params_.downsample_size);
        pnh_.param("use_floor_align", params_.use_floor_align, params_.use_floor_align);
        pnh_.param("floor_z_percentile", params_.floor_z_percentile, params_.floor_z_percentile);
        pnh_.param("floor_ransac_thresh", params_.floor_ransac_thresh, params_.floor_ransac_thresh);
        pnh_.param("floor_ransac_iters", params_.floor_ransac_iters, params_.floor_ransac_iters);
        pnh_.param("voxel_size", params_.voxel_size, params_.voxel_size);
        pnh_.param("n_voxel_min", params_.n_voxel_min, params_.n_voxel_min);
        pnh_.param("n_pillar_min", params_.n_pillar_min, params_.n_pillar_min);
        pnh_.param("use_voxel_filter", params_.use_voxel_filter, params_.use_voxel_filter);
        pnh_.param("voxel_s_thresh", params_.voxel_s_thresh, params_.voxel_s_thresh);
        pnh_.param("simulate_no_ceiling", params_.simulate_no_ceiling, params_.simulate_no_ceiling);
        pnh_.param("ceiling_cut_percentile", params_.ceiling_cut_percentile, params_.ceiling_cut_percentile);
        pnh_.param("use_span_filter", params_.use_span_filter, params_.use_span_filter);
        pnh_.param("H_min_voxel", params_.H_min_voxel, params_.H_min_voxel);
        pnh_.param("fill_ratio_thresh", params_.fill_ratio_thresh, params_.fill_ratio_thresh);
        pnh_.param("horiz_normal_thresh", params_.horiz_normal_thresh, params_.horiz_normal_thresh);
        pnh_.param("use_floor_height_filter", params_.use_floor_height_filter, params_.use_floor_height_filter);
        pnh_.param("floor_height_radius", params_.floor_height_radius, params_.floor_height_radius);
        pnh_.param("floor_height_max", params_.floor_height_max, params_.floor_height_max);
        pnh_.param("use_phi_weight", params_.use_phi_weight, params_.use_phi_weight);
        pnh_.param("phi_mean_thresh", params_.phi_mean_thresh, params_.phi_mean_thresh);
        pnh_.param("phi_std_thresh", params_.phi_std_thresh, params_.phi_std_thresh);
        pnh_.param("s_thresh", params_.s_thresh, params_.s_thresh);
        pnh_.param("use_weight", params_.use_weight, params_.use_weight);
        pnh_.param("n_ref", params_.n_ref, params_.n_ref);
        pnh_.param("mu_thresh", params_.mu_thresh, params_.mu_thresh);
        pnh_.param("sigma_thresh", params_.sigma_thresh, params_.sigma_thresh);
        pnh_.param("use_voxel_corner", params_.use_voxel_corner, params_.use_voxel_corner);
        pnh_.param("corner_multiplier", params_.corner_multiplier, params_.corner_multiplier);
        pnh_.param("use_corner_downsample", params_.use_corner_downsample, params_.use_corner_downsample);
    }

    template <typename T>
    bool readBinaryAt(std::ifstream &file, std::streamoff offset, T &value) const
    {
        file.seekg(offset, std::ios::beg);
        if (!file.good())
            return false;
        file.read(reinterpret_cast<char *>(&value), sizeof(T));
        return file.good();
    }

    bool readLasFirstPointOrigin(const std::string &path,
                                 Eigen::Vector3f &origin) const
    {
        std::ifstream file(path.c_str(), std::ios::binary);
        if (!file.is_open())
            return false;

        char signature[4] = {};
        file.read(signature, 4);
        if (!file.good() || std::strncmp(signature, "LASF", 4) != 0)
            return false;

        uint32_t point_data_offset = 0;
        uint16_t point_record_length = 0;
        double x_scale = 0.0, y_scale = 0.0, z_scale = 0.0;
        double x_offset = 0.0, y_offset = 0.0, z_offset = 0.0;
        if (!readBinaryAt(file, 96, point_data_offset) ||
            !readBinaryAt(file, 105, point_record_length) ||
            !readBinaryAt(file, 131, x_scale) ||
            !readBinaryAt(file, 139, y_scale) ||
            !readBinaryAt(file, 147, z_scale) ||
            !readBinaryAt(file, 155, x_offset) ||
            !readBinaryAt(file, 163, y_offset) ||
            !readBinaryAt(file, 171, z_offset))
        {
            return false;
        }

        if (point_record_length < 14)
            return false;

        int32_t raw_x = 0, raw_y = 0, raw_z = 0;
        if (!readBinaryAt(file, point_data_offset + 0, raw_x) ||
            !readBinaryAt(file, point_data_offset + 4, raw_y) ||
            !readBinaryAt(file, point_data_offset + 8, raw_z))
        {
            return false;
        }

        origin = Eigen::Vector3f(
            static_cast<float>(raw_x * x_scale + x_offset),
            static_cast<float>(raw_y * y_scale + y_offset),
            static_cast<float>(raw_z * z_scale + z_offset));
        return true;
    }

    void configureLocalOrigin()
    {
        std::string mode = local_origin_mode_;
        std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
        local_origin_mode_ = mode;

        if (mode == "none")
        {
            subtract_local_origin_ = false;
            local_origin_ = Eigen::Vector3f::Zero();
            return;
        }

        if (mode == "manual")
        {
            subtract_local_origin_ = true;
            local_origin_ = Eigen::Vector3f(
                static_cast<float>(local_origin_x_),
                static_cast<float>(local_origin_y_),
                static_cast<float>(local_origin_z_));
        }
        else if (mode == "las_first")
        {
            if (origin_las_path_.empty())
                throw std::runtime_error(
                    "local_origin_mode='las_first' requires origin_las_path.");
            if (!readLasFirstPointOrigin(origin_las_path_, local_origin_))
                throw std::runtime_error("Failed to read first LAS point origin: " +
                                         origin_las_path_);
            subtract_local_origin_ = true;
        }
        else
        {
            throw std::runtime_error(
                "local_origin_mode must be one of: none, manual, las_first.");
        }

        ROS_INFO_STREAM("[online_static_feature] subtracting local origin: ["
                        << local_origin_.x() << ", "
                        << local_origin_.y() << ", "
                        << local_origin_.z() << "]");
    }

    static double elapsedMs(const TimePoint &start, const TimePoint &end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

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
            ROS_ERROR_STREAM("[online_static_feature] failed to open latency log: "
                             << latency_log_path_);
            latency_log_enabled_ = false;
            return;
        }

        latency_log_
            << "callback_index,window_index,window_size,msg_stamp_sec,log_stamp_sec,"
            << "input_points,map_points,accumulated_points,"
            << "voxel_count,pillar_count,feature_points,plane_points,corner_points,"
            << "planar_count,corner_count,voxel_min_removed,voxel_dynamic_removed,"
            << "pillar_min_removed,pillar_dynamic_removed,span_removed,fill_removed,"
            << "voxel_corner_points,wall_pillar_count,corner_ds_removed,"
            << "ros_to_pcl_ms,transform_ms,buffer_ms,accumulate_ms,"
            << "floor_align_ms,ceiling_crop_ms,downsample_ms,publish_accumulated_ms,"
            << "build_voxel_pillar_ms,classify_pillars_ms,split_features_ms,"
            << "publish_features_ms,feature_total_ms,"
            << "window_algorithm_ms,window_total_ms\n";
        latency_log_.flush();

        ROS_INFO_STREAM("[online_static_feature] latency log: " << latency_log_path_);
    }

    void writeLatencyLog(const LatencyRecord &r)
    {
        if (!latency_log_enabled_ || !latency_log_.is_open())
            return;

        latency_log_ << std::fixed << std::setprecision(6)
            << r.callback_index << ','
            << r.window_index << ','
            << r.window_size << ','
            << r.msg_stamp_sec << ','
            << r.log_stamp_sec << ','
            << r.input_points << ','
            << r.map_points << ','
            << r.accumulated_points << ','
            << r.feature.voxel_count << ','
            << r.feature.pillar_count << ','
            << r.feature.feature_points << ','
            << r.feature.plane_points << ','
            << r.feature.corner_points << ','
            << r.feature.planar_count << ','
            << r.feature.corner_count << ','
            << r.feature.voxel_min_removed << ','
            << r.feature.voxel_dynamic_removed << ','
            << r.feature.pillar_min_removed << ','
            << r.feature.pillar_dynamic_removed << ','
            << r.feature.span_removed << ','
            << r.feature.fill_removed << ','
            << r.feature.voxel_corner_points << ','
            << r.feature.wall_pillar_count << ','
            << r.feature.corner_ds_removed << ','
            << r.ros_to_pcl_ms << ','
            << r.transform_ms << ','
            << r.buffer_ms << ','
            << r.accumulate_ms << ','
            << r.floor_align_ms << ','
            << r.ceiling_crop_ms << ','
            << r.downsample_ms << ','
            << r.publish_accumulated_ms << ','
            << r.feature.build_voxel_pillar_ms << ','
            << r.feature.classify_pillars_ms << ','
            << r.feature.split_features_ms << ','
            << r.feature.publish_features_ms << ','
            << r.feature.total_ms << ','
            << r.window_algorithm_ms << ','
            << r.window_total_ms << '\n';
        latency_log_.flush();
    }

    void scanPoseCallback(const CloudMsg::ConstPtr &cloud_msg,
                          const PoseMsg::ConstPtr &pose_msg)
    {
        const TimePoint callback_start = SteadyClock::now();
        TimePoint stage_start = callback_start;
        LatencyRecord latency;

        callback_count_++;
        latency.callback_index = callback_count_;
        latency.msg_stamp_sec = cloud_msg->header.stamp.toSec();
        latency.log_stamp_sec = ros::Time::now().toSec();

        pcl::PointCloud<pcl::PointXYZI>::Ptr input_xyzi(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(*cloud_msg, *input_xyzi);
        TimePoint stage_end = SteadyClock::now();
        latency.ros_to_pcl_ms = elapsedMs(stage_start, stage_end);
        latency.input_points = input_xyzi->size();

        if (input_xyzi->empty())
        {
            ROS_WARN_THROTTLE(2.0, "[online_static_feature] empty input cloud");
            return;
        }

        stage_start = SteadyClock::now();
        pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        map_cloud->reserve(input_xyzi->size());

        const Eigen::Affine3f T_map_lidar = poseToAffine(*pose_msg);
        for (const auto &pt : input_xyzi->points)
        {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
                continue;

            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            if (transform_with_pose_)
                p = T_map_lidar * p;
            if (subtract_local_origin_)
                p -= local_origin_;

            pcl::PointXYZ out;
            out.x = p.x();
            out.y = p.y();
            out.z = p.z();
            map_cloud->push_back(out);
        }

        map_cloud->width = static_cast<uint32_t>(map_cloud->size());
        map_cloud->height = 1;
        map_cloud->is_dense = true;
        stage_end = SteadyClock::now();
        latency.transform_ms = elapsedMs(stage_start, stage_end);
        latency.map_points = map_cloud->size();

        stage_start = SteadyClock::now();
        frame_buffer_.push_back(map_cloud);
        while (static_cast<int>(frame_buffer_.size()) > window_frames_)
            frame_buffer_.pop_front();
        stage_end = SteadyClock::now();
        latency.buffer_ms = elapsedMs(stage_start, stage_end);

        if (callback_count_ % publish_every_n_frames_ != 0)
            return;

        processed_window_count_++;
        latency.window_index = processed_window_count_;
        latency.window_size = static_cast<int>(frame_buffer_.size());

        stage_start = SteadyClock::now();
        pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated = makeAccumulatedCloud();
        stage_end = SteadyClock::now();
        latency.accumulate_ms = elapsedMs(stage_start, stage_end);

        if (params_.use_floor_align)
        {
            stage_start = SteadyClock::now();
            Eigen::Matrix3f floor_R = Eigen::Matrix3f::Identity();
            float floor_z_ref = 0.0f;
            accumulated = correctFloorAlignment(accumulated, params_, floor_R, floor_z_ref);
            stage_end = SteadyClock::now();
            latency.floor_align_ms = elapsedMs(stage_start, stage_end);
        }
        if (params_.simulate_no_ceiling)
        {
            stage_start = SteadyClock::now();
            accumulated = cropPointsAboveZ(accumulated, params_.ceiling_cut_percentile);
            stage_end = SteadyClock::now();
            latency.ceiling_crop_ms = elapsedMs(stage_start, stage_end);
        }
        if (params_.use_downsample)
        {
            stage_start = SteadyClock::now();
            accumulated = downsampleCloud(accumulated);
            stage_end = SteadyClock::now();
            latency.downsample_ms = elapsedMs(stage_start, stage_end);
        }

        latency.accumulated_points = accumulated->size();
        stage_start = SteadyClock::now();
        publishXYZCloud(accumulated, accumulated_pub_, cloud_msg->header.stamp);
        stage_end = SteadyClock::now();
        latency.publish_accumulated_ms = elapsedMs(stage_start, stage_end);

        extractAndPublishFeatures(accumulated, cloud_msg->header.stamp, latency.feature);

        latency.window_algorithm_ms =
            latency.accumulate_ms +
            latency.floor_align_ms +
            latency.ceiling_crop_ms +
            latency.downsample_ms +
            latency.feature.build_voxel_pillar_ms +
            latency.feature.classify_pillars_ms +
            latency.feature.split_features_ms;
        latency.window_total_ms =
            latency.window_algorithm_ms +
            latency.publish_accumulated_ms +
            latency.feature.publish_features_ms;
        writeLatencyLog(latency);
    }

    Eigen::Affine3f poseToAffine(const PoseMsg &pose_msg) const
    {
        const auto &p = pose_msg.pose.position;
        const auto &q_msg = pose_msg.pose.orientation;
        Eigen::Quaternionf q(
            static_cast<float>(q_msg.w),
            static_cast<float>(q_msg.x),
            static_cast<float>(q_msg.y),
            static_cast<float>(q_msg.z));
        if (q.norm() < 1e-6f)
            q = Eigen::Quaternionf::Identity();
        else
            q.normalize();

        Eigen::Affine3f T = Eigen::Affine3f::Identity();
        T.translation() = Eigen::Vector3f(
            static_cast<float>(p.x),
            static_cast<float>(p.y),
            static_cast<float>(p.z));
        T.linear() = q.toRotationMatrix();
        return T;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr makeAccumulatedCloud() const
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated(new pcl::PointCloud<pcl::PointXYZ>());
        size_t total_points = 0;
        for (const auto &frame : frame_buffer_)
            total_points += frame->size();
        accumulated->reserve(total_points);

        for (const auto &frame : frame_buffer_)
            *accumulated += *frame;

        accumulated->width = static_cast<uint32_t>(accumulated->size());
        accumulated->height = 1;
        accumulated->is_dense = true;
        return accumulated;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampleCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) const
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr ds(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(params_.downsample_size,
                       params_.downsample_size,
                       params_.downsample_size);
        vg.filter(*ds);
        return ds;
    }

    void extractAndPublishFeatures(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                   const ros::Time &stamp,
                                   FeatureTiming &timing)
    {
        const TimePoint feature_start = SteadyClock::now();
        TimePoint stage_start = feature_start;

        std::unordered_map<VoxelKey, VoxelPoints, VoxelKeyHash> voxel_map;
        std::unordered_map<PillarKey, PillarData, PillarKeyHash> pillar_map;
        int voxel_min_removed = 0;
        buildVoxelAndPillarMap(cloud, params_, voxel_map, pillar_map, voxel_min_removed);
        TimePoint stage_end = SteadyClock::now();
        timing.build_voxel_pillar_ms = elapsedMs(stage_start, stage_end);
        timing.voxel_count = voxel_map.size();
        timing.pillar_count = pillar_map.size();
        timing.voxel_min_removed = voxel_min_removed;

        int planar_cnt = 0;
        int corner_cnt = 0;
        int voxel_dyn_cnt = 0;
        int pillar_min_cnt = 0;
        int pillar_dyn_cnt = 0;
        int span_dyn_cnt = 0;
        int fill_dyn_cnt = 0;
        int voxel_corner_cnt = 0;
        int wall_pillar_cnt = 0;
        int corner_ds_removed = 0;

        stage_start = SteadyClock::now();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored = classifyAllPillars(
            pillar_map,
            params_,
            0.0f,
            planar_cnt,
            corner_cnt,
            voxel_dyn_cnt,
            pillar_min_cnt,
            pillar_dyn_cnt,
            span_dyn_cnt,
            fill_dyn_cnt,
            voxel_corner_cnt,
            wall_pillar_cnt,
            corner_ds_removed);
        stage_end = SteadyClock::now();
        timing.classify_pillars_ms = elapsedMs(stage_start, stage_end);
        timing.feature_points = colored->size();
        timing.planar_count = planar_cnt;
        timing.corner_count = corner_cnt;
        timing.voxel_dynamic_removed = voxel_dyn_cnt;
        timing.pillar_min_removed = pillar_min_cnt;
        timing.pillar_dynamic_removed = pillar_dyn_cnt;
        timing.span_removed = span_dyn_cnt;
        timing.fill_removed = fill_dyn_cnt;
        timing.voxel_corner_points = voxel_corner_cnt;
        timing.wall_pillar_count = wall_pillar_cnt;
        timing.corner_ds_removed = corner_ds_removed;

        stage_start = SteadyClock::now();
        pcl::PointCloud<pcl::PointXYZ>::Ptr planes(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr corners(new pcl::PointCloud<pcl::PointXYZ>());
        planes->reserve(static_cast<size_t>(planar_cnt));
        corners->reserve(static_cast<size_t>(corner_cnt));

        for (const auto &pt : colored->points)
        {
            pcl::PointXYZ out;
            out.x = pt.x;
            out.y = pt.y;
            out.z = pt.z;

            const bool is_planar = (pt.r > 200 && pt.g > 200 && pt.b > 200);
            if (is_planar)
                planes->push_back(out);
            else
                corners->push_back(out);
        }

        planes->width = static_cast<uint32_t>(planes->size());
        planes->height = 1;
        planes->is_dense = true;
        corners->width = static_cast<uint32_t>(corners->size());
        corners->height = 1;
        corners->is_dense = true;
        stage_end = SteadyClock::now();
        timing.split_features_ms = elapsedMs(stage_start, stage_end);
        timing.plane_points = planes->size();
        timing.corner_points = corners->size();

        stage_start = SteadyClock::now();
        publishXYZRGBCloud(colored, colored_feature_pub_, stamp);
        publishXYZCloud(planes, plane_feature_pub_, stamp);
        publishXYZCloud(corners, corner_feature_pub_, stamp);
        stage_end = SteadyClock::now();
        timing.publish_features_ms = elapsedMs(stage_start, stage_end);
        timing.total_ms = elapsedMs(feature_start, stage_end);

        updateFeatureHistory(colored, stamp);

        ROS_INFO_THROTTLE(
            1.0,
            "[online_static_feature] frames=%zu accumulated=%zu features=%zu planar=%d corner=%d voxel_min_removed=%d pillar_dyn=%d span_removed=%d fill_removed=%d",
            frame_buffer_.size(),
            cloud->size(),
            colored->size(),
            planar_cnt,
            corner_cnt,
            voxel_min_removed,
            pillar_dyn_cnt,
            span_dyn_cnt,
            fill_dyn_cnt);
    }

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
                    ROS_ERROR_STREAM("[online_static_feature] mkdir failed: "
                                     << current << " (" << std::strerror(errno) << ")");
                    return false;
                }
            }
            if (next == std::string::npos)
                break;
            start = next + 1;
        }
        return true;
    }

    void updateFeatureHistory(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &colored,
                              const ros::Time &stamp)
    {
        processed_feature_frames_++;

        if (!publish_feature_history_ && !save_on_shutdown_pcd_)
            return;

        *feature_history_ += *colored;
        if (max_feature_history_points_ > 0 &&
            static_cast<int>(feature_history_->size()) > max_feature_history_points_)
        {
            const size_t excess = feature_history_->size() -
                static_cast<size_t>(max_feature_history_points_);
            feature_history_->points.erase(
                feature_history_->points.begin(),
                feature_history_->points.begin() + static_cast<std::ptrdiff_t>(excess));
        }
        feature_history_->width = static_cast<uint32_t>(feature_history_->size());
        feature_history_->height = 1;
        feature_history_->is_dense = true;

        if (publish_feature_history_)
            publishXYZRGBCloud(feature_history_, feature_history_pub_, stamp);
    }

    bool saveFeatureHistoryPcd(const std::string &reason)
    {
        if (!feature_history_ || feature_history_->empty())
        {
            ROS_WARN_STREAM("[online_static_feature] skip " << reason
                            << " PCD save: feature history is empty");
            return false;
        }

        if (!ensureParentDirectory(save_feature_history_path_))
            return false;

        if (pcl::io::savePCDFileBinary(save_feature_history_path_, *feature_history_) == 0)
        {
            ROS_INFO_STREAM("[online_static_feature] saved " << reason
                            << " feature history PCD: "
                            << save_feature_history_path_
                            << " (" << feature_history_->size() << " points)");
            return true;
        }

        ROS_ERROR_STREAM("[online_static_feature] failed to save " << reason
                         << " feature history PCD: "
                         << save_feature_history_path_);
        return false;
    }

    void publishXYZCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                         const ros::Publisher &pub,
                         const ros::Time &stamp) const
    {
        if (!pub)
            return;
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.stamp = stamp;
        msg.header.frame_id = map_frame_;
        pub.publish(msg);
    }

    void publishXYZRGBCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                            const ros::Publisher &pub,
                            const ros::Time &stamp) const
    {
        if (!pub)
            return;
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.stamp = stamp;
        msg.header.frame_id = map_frame_;
        pub.publish(msg);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "online_static_feature_node");
    OnlineStaticFeatureNode node;
    ros::spin();
    return 0;
}
