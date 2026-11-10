#include "pillar_pca.hpp"
#include "types.hpp"
#include "voxel_map.hpp"

#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/common/transforms.h>
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
    // 초기화 관련 함수
    OnlineStaticFeatureNode()
        : nh_(),        // NodeHandle
          pnh_("~"),    // private NodeHandle
          cloud_sub_(),
          pose_sub_(),
          sync_(SyncPolicy(20), cloud_sub_, pose_sub_)
    {
        loadRosParams();         // ROS 파라미터 읽기
        configureLocalOrigin();  // local origin 설정
        initLatencyLog();        // latency CSV chrlghk

        // 입력 2개(cloud / pose)의 subscriber 각각 연결
        cloud_sub_.subscribe(nh_, cloud_topic_, 1); // cloud_topic : /scan_matching_accumulator_node/dense_window_cloud
        pose_sub_.subscribe(nh_, pose_topic_, 10);  // pose_topic  : /scan_matching_accumulator_node/scan_matching_pose

        // (옵션) map_localizer가 발행하는 T_map_odom. extract_in_map_frame=true일 때
        // window를 map 좌표계로 옮겨 voxel/pillar 격자를 구성하는 데 쓴다.
        if (extract_frame_ != "odom")
            map_odom_sub_ = nh_.subscribe(map_odom_topic_, 10,
                                          &OnlineStaticFeatureNode::mapOdomCallback, this);

        // ApproximateTime synchronizer 설정
        sync_.setMaxIntervalDuration(ros::Duration(sync_slop_sec_));
        // cloud와 pose쌍을 찾아 하나의 콜백으로 전달하여 싱크 맞춰주기
        sync_.registerCallback(boost::bind(
            &OnlineStaticFeatureNode::scanPoseCallback, this, _1, _2)); // _1 : cloud, _2 : pose
        
        // 출력 publisher 생성
        accumulated_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            accumulated_topic_, 1, false);
        colored_feature_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            colored_feature_topic_, 1, false);
        plane_feature_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            plane_feature_topic_, 1, false);
        corner_feature_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            corner_feature_topic_, 1, false);
        static_feature_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            static_feature_topic_, 1, false);
        feature_history_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
            feature_history_topic_, 1, true);

        ROS_INFO_STREAM("[online_static_feature] cloud_topic=" << cloud_topic_
                        << ", pose_topic=" << pose_topic_
                        << ", window_frames=" << window_frames_
                        << ", transform_with_pose=" << transform_with_pose_
                        << ", extract_frame=" << extract_frame_
                        << ", local_origin_mode=" << local_origin_mode_);
    }

    ~OnlineStaticFeatureNode() // 노드 종료 시 누적된 static feature map 저장
    {
        if (save_on_shutdown_pcd_)
            saveFeatureHistoryPcd("shutdown");
    }

private:
    using CloudMsg = sensor_msgs::PointCloud2;
    using PoseMsg = geometry_msgs::PoseStamped;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<CloudMsg, PoseMsg>; // cloud와 pose를 동기화
    using SteadyClock = std::chrono::steady_clock;
    using TimePoint = SteadyClock::time_point;

    // 특징 추출 단계의 시간과 통계를 보관
    struct FeatureTiming
    {
        double build_voxel_pillar_ms = 0.0; // voxel map 생성 시간
        // build_voxel_pillar 하위 분해 (병목 분석용)
        double bvp_step1_assign_ms = 0.0;   // 포인트 → voxel 해시 삽입
        double bvp_step1_merge_ms = 0.0;    // shard → 단일 map splice (직렬이면 0)
        double bvp_step2_pillar_ms = 0.0;   // voxel → pillar 연결
        double classify_pillars_ms = 0.0;
        // classify_pillars 하위 분해. 둘의 합과 classify_pillars_ms의 차이가
        // 필터([1]~[3])와 unordered_map 순회 비용이다.
        double cls_pillar_math_ms = 0.0;    // computePillarPCA + NormalConsistency
        double cls_emit_ms = 0.0;           // 출력 클라우드 적재
        double cls_voxel_pca_ms = 0.0;      // eigenDecompose 총합 (voxel PCA)
        double cls_phi_ms = 0.0;            // Stage 3 φ 통계
        long   cls_voxel_pca_cnt = 0;       // eigenDecompose 호출 횟수
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

    // callback 전체에 대한 시간과 포인트 수를 보관
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
        double map_frame_ms = 0.0;      // extract_in_map_frame: window에 T_map_odom 적용 시간
        double window_algorithm_ms = 0.0;
        double window_total_ms = 0.0;

        FeatureTiming feature;
    };

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    std::string cloud_topic_ = "/Pointcloud2";
    std::string pose_topic_ = "/lidar_pose";
    // extract_in_map_frame=true일 때 구독하는 T_map_odom (map_localizer 출력)
    std::string map_odom_topic_ = "/map_localizer/map_odom_pose";
    std::string map_frame_ = "map";
    std::string accumulated_topic_ = "accumulated_cloud";
    std::string colored_feature_topic_ = "static_features_colored";
    std::string plane_feature_topic_ = "scan_plane_features";
    std::string corner_feature_topic_ = "scan_corner_features";
    // use_feature_classification=0일 때 쓰는 단일 정적 포인트 토픽
    std::string static_feature_topic_ = "scan_static_features";
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
    // true: 최신 T_map_odom을 window 전체에 곱해 map 좌표계에서 추출하고,
    //       출력은 다시 odom 좌표계로 되돌린다. false: 기존대로 odom 좌표계에서 추출.
    bool extract_in_map_frame_ = false;   // 구 파라미터. true면 extract_frame=map_live 와 같다.
    // extract_frame: f_MAV(voxel/pillar 격자)를 어느 좌표계에서 돌릴지.
    //   odom     : odom(=첫 스캔 센서) 좌표계. 격자가 장면에 고정. 수직축 = 첫 센서 z. (기본)
    //   map_init : 처음 받은 T_map_odom(= 초기 전역자세 ^W T_0)을 고정해서 곱한다.
    //              격자가 장면에 고정되면서 수직축·격자 위상은 map 기준. 초기 roll/pitch 가정 제거.
    //   map_live : 매 프레임 최신 T_map_odom을 곱한다. 격자가 C의 흔들림만큼 움직인다.
    //              (노은역 full-bag A/B에서 fitness 악화 → ablation 용으로만 남김)
    //   odom_level : 최신 T_map_odom의 roll·pitch만 적용(yaw·translation 제거). 격자 원점·heading·
    //              xy 위상은 odom 그대로 두고 수직축만 map(중력)에 맞춘다. 기울기 효과만 분리하는 조건.
    // 어느 모드든 출력은 같은 변환의 역으로 odom 좌표계로 되돌려 발행한다.
    std::string extract_frame_ = "odom";
    bool has_map_odom_ = false;
    Eigen::Affine3f T_map_odom_latest_ = Eigen::Affine3f::Identity();
    Eigen::Affine3f T_map_odom_first_ = Eigen::Affine3f::Identity();
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
    ros::Subscriber map_odom_sub_;

    ros::Publisher accumulated_pub_;
    ros::Publisher colored_feature_pub_;
    ros::Publisher plane_feature_pub_;
    ros::Publisher corner_feature_pub_;
    ros::Publisher static_feature_pub_;
    ros::Publisher feature_history_pub_;

    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> frame_buffer_; // 최근 cloud들을 저장하는 sliding window
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr feature_history_{
        new pcl::PointCloud<pcl::PointXYZRGB>()};
    std::ofstream latency_log_;

    void loadRosParams() // ROS 파라미터 읽어오기
    {
        pnh_.param<std::string>("cloud_topic", cloud_topic_, cloud_topic_);
        pnh_.param<std::string>("pose_topic", pose_topic_, pose_topic_);
        pnh_.param<std::string>("map_frame", map_frame_, map_frame_);
        pnh_.param<std::string>("accumulated_topic", accumulated_topic_, accumulated_topic_);
        pnh_.param<std::string>("colored_feature_topic", colored_feature_topic_, colored_feature_topic_);
        pnh_.param<std::string>("plane_feature_topic", plane_feature_topic_, plane_feature_topic_);
        pnh_.param<std::string>("corner_feature_topic", corner_feature_topic_, corner_feature_topic_);
        pnh_.param<std::string>("static_feature_topic", static_feature_topic_, static_feature_topic_);
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
        pnh_.param("extract_in_map_frame", extract_in_map_frame_, extract_in_map_frame_);
        pnh_.param<std::string>("extract_frame", extract_frame_, extract_frame_);
        pnh_.param<std::string>("map_odom_topic", map_odom_topic_, map_odom_topic_);
        if (extract_in_map_frame_ && extract_frame_ == "odom")
            extract_frame_ = "map_live";   // 구 bool 파라미터와의 호환
        if (extract_frame_ != "odom" && extract_frame_ != "map_init" && extract_frame_ != "map_live" &&
            extract_frame_ != "odom_level")
        {
            ROS_WARN_STREAM("[online_static_feature] unknown extract_frame='" << extract_frame_
                            << "', falling back to 'odom'");
            extract_frame_ = "odom";
        }
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
        params_.use_feature_classification = 1;

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
        pnh_.param("use_feature_classification", params_.use_feature_classification,
                   params_.use_feature_classification);
        pnh_.param("classify_num_threads", params_.classify_num_threads,
                   params_.classify_num_threads);
        pnh_.param("voxel_num_threads", params_.voxel_num_threads,
                   params_.voxel_num_threads);
    }

    template <typename T>
    // 좌표 원점 관련 함수 - las 파일 헤더와 첫번째 포인트 좌표를 읽을 때 사용
    // 바이너리 파일의 특정 byte offset에서 원하는 타입을 읽는 템플릿 함수임
    bool readBinaryAt(std::ifstream &file, std::streamoff offset, T &value) const
    {
        file.seekg(offset, std::ios::beg);
        if (!file.good())
            return false;
        file.read(reinterpret_cast<char *>(&value), sizeof(T));
        return file.good();
    }

    // LAS 파일을 직접 열어 여러 정보들을 읽어오는 함수
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

        // 데이터 읽어오기
        uint32_t point_data_offset = 0;
        uint16_t point_record_length = 0;
        double x_scale = 0.0, y_scale = 0.0, z_scale = 0.0;
        double x_offset = 0.0, y_offset = 0.0, z_offset = 0.0;
        if (!readBinaryAt(file, 96, point_data_offset) ||       // point data 시작 위치
            !readBinaryAt(file, 105, point_record_length) ||    // point record 길이
            !readBinaryAt(file, 131, x_scale) ||                // x, y, z scale
            !readBinaryAt(file, 139, y_scale) ||
            !readBinaryAt(file, 147, z_scale) ||
            !readBinaryAt(file, 155, x_offset) ||               // x, y, z offset
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

        // 포인트의 실제 좌표 복원
        origin = Eigen::Vector3f(
            static_cast<float>(raw_x * x_scale + x_offset),
            static_cast<float>(raw_y * y_scale + y_offset),
            static_cast<float>(raw_z * z_scale + z_offset));
        return true;
    }

    // LAS - 지원하는 원점 모드 3가지
    void configureLocalOrigin()
    {
        std::string mode = local_origin_mode_;
        std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
        local_origin_mode_ = mode;

        // none: 좌표값 그대로 사용
        if (mode == "none")
        {
            subtract_local_origin_ = false;
            local_origin_ = Eigen::Vector3f::Zero();
            return;
        }

        // manual: 파라미터로 받은 (x, y, z)를 빼줌
        if (mode == "manual")
        {
            subtract_local_origin_ = true;
            local_origin_ = Eigen::Vector3f(
                static_cast<float>(local_origin_x_),
                static_cast<float>(local_origin_y_),
                static_cast<float>(local_origin_z_));
        }

        // LAS 파일의 첫번째 포인트가 원점이 되도록 함
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

    // 두 시점 차이를 밀리 초(ms)로 변환 - 로그 기록용
    static double elapsedMs(const TimePoint &start, const TimePoint &end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    // 상위 디렉토리를 만들고 CSV 파일을 연 뒤 헤더를 기록
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
            << "build_voxel_pillar_ms,bvp_step1_assign_ms,bvp_step1_merge_ms,"
            << "bvp_step2_pillar_ms,"
            << "classify_pillars_ms,cls_pillar_math_ms,cls_emit_ms,"
            << "cls_voxel_pca_ms,cls_voxel_pca_cnt,cls_phi_ms,split_features_ms,"
            << "publish_features_ms,feature_total_ms,"
            << "window_algorithm_ms,window_total_ms,map_frame_ms\n";
        latency_log_.flush();

        ROS_INFO_STREAM("[online_static_feature] latency log: " << latency_log_path_);
    }

    // LatencyRecord와 그 안의 FeatureTiming을 하나의 행으로 저장함
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
            << r.feature.bvp_step1_assign_ms << ','
            << r.feature.bvp_step1_merge_ms << ','
            << r.feature.bvp_step2_pillar_ms << ','
            << r.feature.classify_pillars_ms << ','
            << r.feature.cls_pillar_math_ms << ','
            << r.feature.cls_emit_ms << ','
            << r.feature.cls_voxel_pca_ms << ','
            << r.feature.cls_voxel_pca_cnt << ','
            << r.feature.cls_phi_ms << ','
            << r.feature.split_features_ms << ','
            << r.feature.publish_features_ms << ','
            << r.feature.total_ms << ','
            << r.window_algorithm_ms << ','
            << r.window_total_ms << ','
            << r.map_frame_ms << '\n';
        latency_log_.flush();
    }

    // 핵심 콜백 함수
    // 입력 : dense_window_cloud (=accumulator에서 발행한 PointXYZI cloud), odom 좌표계임
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

        // 1. ROS cloud를 PCL로 변환
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

        // 2. 출력되는 map_cloud 좌표계(동일하게 odom)의 PointXYZ cloud 생성 - Intensity는 이후 사용하지 않으므로 제거함
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
                p -= local_origin_; // 원점 기준 정렬

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

        // 3. online sliding window 갱신
        stage_start = SteadyClock::now();
        frame_buffer_.push_back(map_cloud);
        while (static_cast<int>(frame_buffer_.size()) > window_frames_) // 최근 window_frames 개의 입력 frame만 유지
            frame_buffer_.pop_front();
        stage_end = SteadyClock::now();
        latency.buffer_ms = elapsedMs(stage_start, stage_end);

        // 4. 처리 주기 확인
        // 값이 3이면 cloud는 계속 buffer에 들어가지만, 특징 추출은 3 callback마다 한번 수행
        if (callback_count_ % publish_every_n_frames_ != 0)
            return;

        processed_window_count_++;
        latency.window_index = processed_window_count_;
        latency.window_size = static_cast<int>(frame_buffer_.size());

        // 5. 버퍼 내 모든 포인트를 하나의 cloud로 합쳐줌
        // 이미 각 입력 포인트 cloud는 동일한 odom 좌표계에 있으므로, pose 변환 불필요
        stage_start = SteadyClock::now();
        pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated = makeAccumulatedCloud();
        stage_end = SteadyClock::now();
        latency.accumulate_ms = elapsedMs(stage_start, stage_end);

        // 6. yaml 옵션에 따른 전처리
        if (params_.use_floor_align) // 바닥 정렬
        {
            stage_start = SteadyClock::now();
            Eigen::Matrix3f floor_R = Eigen::Matrix3f::Identity();
            float floor_z_ref = 0.0f;
            accumulated = correctFloorAlignment(accumulated, params_, floor_R, floor_z_ref);
            stage_end = SteadyClock::now();
            latency.floor_align_ms = elapsedMs(stage_start, stage_end);
        }
        if (params_.simulate_no_ceiling) // 천장 제거 시뮬레이션
        {
            stage_start = SteadyClock::now();
            accumulated = cropPointsAboveZ(accumulated, params_.ceiling_cut_percentile);
            stage_end = SteadyClock::now();
            latency.ceiling_crop_ms = elapsedMs(stage_start, stage_end);
        }
        if (params_.use_downsample) // VoxelGrid 다운샘플링
        {
            stage_start = SteadyClock::now();
            accumulated = downsampleCloud(accumulated);
            stage_end = SteadyClock::now();
            latency.downsample_ms = elapsedMs(stage_start, stage_end);
        }

        // 7. 전처리된 acuumulated cloud 발행
        latency.accumulated_points = accumulated->size();
        stage_start = SteadyClock::now();
        // 포인트 발행
        // dense_window_cloud를 입력으로 받아서, 선택적 전처리를 적용한 accumulated_cloud로 발행함
        publishXYZCloud(accumulated, accumulated_pub_, cloud_msg->header.stamp);
        stage_end = SteadyClock::now();
        latency.publish_accumulated_ms = elapsedMs(stage_start, stage_end);

        // 8. (옵션) 추출 좌표계 선택
        // extract_in_map_frame=true: 최신 T_map_odom(C)을 window 전체에 똑같이 곱해
        // map 좌표계에서 voxel/pillar 격자를 구성한다. window 내부 상대 배치는 그대로다.
        // 추출 결과는 C^-1로 되돌려 odom 좌표계로 발행하므로 다른 노드는 바뀌지 않는다.
        // (정합은 강체변환에 공변이라 localizer가 C를 다시 곱해도 같은 문제를 푼다.)
        // 아직 C를 못 받았으면(첫 프레임) 기존대로 odom 좌표계에서 추출한다.
        // accumulated_cloud는 위에서 이미 odom 좌표계로 발행됐다.
        const Eigen::Affine3f *T_back = nullptr;
        Eigen::Affine3f T_odom_map = Eigen::Affine3f::Identity();
        if (extract_frame_ != "odom" && has_map_odom_)
        {
            stage_start = SteadyClock::now();
            Eigen::Affine3f T_extract =
                (extract_frame_ == "map_init") ? T_map_odom_first_ : T_map_odom_latest_;
            if (extract_frame_ == "odom_level")
            {
                // R_level = Rz(yaw)^T · R(C): yaw를 빼서 roll·pitch(수직축 정렬)만 남긴다. translation 0.
                const Eigen::Matrix3f R = T_map_odom_latest_.linear();
                const float yaw = std::atan2(R(1, 0), R(0, 0));
                T_extract = Eigen::Affine3f(Eigen::AngleAxisf(-yaw, Eigen::Vector3f::UnitZ()) * R);
            }
            pcl::transformPointCloud(*accumulated, *accumulated, T_extract);
            T_odom_map = T_extract.inverse();
            T_back = &T_odom_map;
            stage_end = SteadyClock::now();
            latency.map_frame_ms = elapsedMs(stage_start, stage_end);
        }

        // 9. 특징점 추출
        extractAndPublishFeatures(accumulated, cloud_msg->header.stamp, latency.feature, T_back);

        latency.window_algorithm_ms =
            latency.accumulate_ms +
            latency.floor_align_ms +
            latency.ceiling_crop_ms +
            latency.downsample_ms +
            latency.map_frame_ms +
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

    // map_localizer의 T_map_odom 수신. 가장 최근 값만 보관한다.
    void mapOdomCallback(const PoseMsg::ConstPtr &msg)
    {
        T_map_odom_latest_ = poseToAffine(*msg);
        if (!has_map_odom_)
        {
            has_map_odom_ = true;
            T_map_odom_first_ = T_map_odom_latest_;   // map_init 모드가 고정해서 쓰는 ^W T_0
            const Eigen::Vector3f t = T_map_odom_first_.translation();
            ROS_INFO_STREAM("[online_static_feature] first T_map_odom received from "
                            << map_odom_topic_ << " (t=" << t.x() << ", " << t.y() << ", " << t.z()
                            << "); extract_frame=" << extract_frame_ << " active from now on");
        }
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

    // 정적 특징 추출 함수
    // T_back: null이 아니면 추출 결과(colored)에 곱해 출력 좌표계로 되돌린다
    //         (extract_in_map_frame일 때 T_odom_map). 입력 cloud 좌표계에서 격자를 만든다.
    void extractAndPublishFeatures(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                   const ros::Time &stamp,
                                   FeatureTiming &timing,
                                   const Eigen::Affine3f *T_back = nullptr)
    {
        const TimePoint feature_start = SteadyClock::now();
        TimePoint stage_start = feature_start;

        std::unordered_map<VoxelKey, VoxelPoints, VoxelKeyHash> voxel_map;
        std::unordered_map<PillarKey, PillarData, PillarKeyHash> pillar_map;
        int voxel_min_removed = 0;

        // 1. Voxel map & pillar map 생성 - 실구현: voxel_map.cpp 참조
        BuildTiming build_timing;
        buildVoxelAndPillarMap(cloud, params_, voxel_map, pillar_map, voxel_min_removed,
                               &build_timing);
        TimePoint stage_end = SteadyClock::now();
        timing.build_voxel_pillar_ms = elapsedMs(stage_start, stage_end);
        timing.bvp_step1_assign_ms = build_timing.step1_assign_ms;
        timing.bvp_step1_merge_ms = build_timing.step1_merge_ms;
        timing.bvp_step2_pillar_ms = build_timing.step2_pillar_ms;
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

        ClassifyTiming classify_timing;
        stage_start = SteadyClock::now();
        // 2. 모든 pillar를 분류 - pillar_pca.cpp 참조
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
            corner_ds_removed,
            &classify_timing);
        stage_end = SteadyClock::now();
        timing.classify_pillars_ms = elapsedMs(stage_start, stage_end);
        timing.cls_pillar_math_ms = classify_timing.pillar_math_ms;
        timing.cls_emit_ms = classify_timing.emit_ms;
        timing.cls_voxel_pca_ms = classify_timing.voxel_pca_ms;
        timing.cls_phi_ms = classify_timing.phi_ms;
        timing.cls_voxel_pca_cnt = classify_timing.voxel_pca_cnt;
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
        // 추출 좌표계가 map(extract_in_map_frame)이었다면 출력은 odom으로 되돌린다.
        // planes/corners/feature_history 모두 colored에서 파생되므로 여기 한 번이면 된다.
        // (split_features_ms에 포함)
        if (T_back)
            pcl::transformPointCloud(*colored, *colored, *T_back);
        pcl::PointCloud<pcl::PointXYZ>::Ptr planes(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr corners(new pcl::PointCloud<pcl::PointXYZ>());

        // 3. RGB cloud를 plane / corner로 특징점 라벨 분리
        // plane : 흰색 / corner : 파랑 or 빨강
        if (params_.use_feature_classification)
        {
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
        }
        else
        {
            // 분류를 쓰지 않으므로 색 기준 분리 없이 그대로 옮긴다.
            // corners는 비워 두고 planes만 단일 토픽으로 발행한다.
            planes->reserve(colored->size());
            for (const auto &pt : colored->points)
            {
                pcl::PointXYZ out;
                out.x = pt.x;
                out.y = pt.y;
                out.z = pt.z;
                planes->push_back(out);
            }
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
        if (params_.use_feature_classification)
        {
            publishXYZRGBCloud(colored, colored_feature_pub_, stamp); // 발행 토픽 : static_features_colored
            publishXYZCloud(planes, plane_feature_pub_, stamp);       // 빌헹 토픽 : scan_plane_features
            publishXYZCloud(corners, corner_feature_pub_, stamp);     // 발행 토픽 : scan_corner_features
        }
        else
        {
            // colored(XYZRGB)와 plane/corner 분리 발행을 건너뛴다.
            publishXYZCloud(planes, static_feature_pub_, stamp);      // 발행 토픽 : scan_static_features
        }
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

    // 각 window에서 추출한 colored feature를 누적함 -> 종료 시 static feature PCD 저장을 위함
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

    // 종료 시 누적된 feature_history_를 binary PCD로 저장
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

    // PointXYZ cloud를 PointCloud2로 변환
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

    // PointXYZRGB cloud를 PointCloud2로 변환
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
