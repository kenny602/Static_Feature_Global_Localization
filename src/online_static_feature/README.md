# online_static_feature

ROS1 online static feature experiment package.

This package subscribes to the LAS replay topics from `las_pcl`:

- `/Pointcloud2` (`sensor_msgs/PointCloud2`)
- `/lidar_pose` (`geometry_msgs/PoseStamped`)

Each incoming sparse scan is transformed into the global map frame with
`/lidar_pose`, then the first LAS point origin is subtracted so the result
matches PCD maps exported with `local_origin_mode=first`. The latest N frames
are accumulated, and the existing voxel/pillar PCA feature extraction logic
from `generalize_static_feature_ws` is applied to the accumulated cloud.

## Topics

Published topics are private to the node namespace by default:

- `/online_static_feature_node/accumulated_cloud`
- `/online_static_feature_node/static_features_colored`
- `/online_static_feature_node/scan_plane_features`
- `/online_static_feature_node/scan_corner_features`
- `/online_static_feature_node/feature_map_colored`

`static_features_colored` uses the existing color convention:

- white: planar feature
- blue: corner pillar feature
- red: voxel corner feature

## Build

```bash
cd /home/rain/online_static_feature_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Run

In one terminal, run the LAS replay publisher:

```bash
cd /home/rain/las_pcl_ROS1
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch las_pcl pcl_pub.launch
```

In another terminal, run online feature extraction:

```bash
cd /home/rain/online_static_feature_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch online_static_feature online_static_feature.launch
```

Try different accumulation windows:

```bash
roslaunch online_static_feature online_static_feature.launch window_frames:=5
```

If the input cloud is already in the map frame, disable pose transform:

```bash
roslaunch online_static_feature online_static_feature.launch transform_with_pose:=false
```

Save the accumulated colored feature history as a PCD on Ctrl+C:

```bash
roslaunch online_static_feature online_static_feature.launch save_on_shutdown_pcd:=true
```

The default output path is:

```bash
/home/rain/online_static_feature_ws/output/online_static_features.pcd
```

Per-window latency is written as CSV while the node runs:

```bash
/home/rain/online_static_feature_ws/output/online_static_feature_latency.csv
```

Each row reports only the current sliding-window latency. Use
`window_algorithm_ms` for processing-only time and `window_total_ms` for the
current window's processing plus current-window publish time.

Override or disable it with:

```bash
roslaunch online_static_feature online_static_feature.launch latency_log_path:=/tmp/latency.csv
roslaunch online_static_feature online_static_feature.launch latency_log_enabled:=false
```

If you want UTM/GIS absolute coordinates instead of the local PCD-map
coordinate system:

```bash
roslaunch online_static_feature online_static_feature.launch local_origin_mode:=none
```
