# Static_Feature_Global_Localization

Global Localization with static feature extraction module based on the geometrical information of LiDAR point cloud

동적 상황에서의 localization을 위한, 라이다 type에 따른 dependency가 없는 포인트 분포 기반 static feature point extraction 및 이를 이용한 scan-to-map global localization.

## Nodes

| Node | 역할 |
| --- | --- |
| `scan_matching_accumulator_node` | raw LiDAR 정합으로 `T_odom_lidar` 추정, dense window cloud 발행 |
| `online_static_feature_node` | voxel/pillar PCA 기반 static feature(plane/corner) 온라인 추출 |
| `pcd_window_feature_batch_node` | 저장된 PCD 시퀀스 오프라인 배치 처리 |
| `map_localizer_node` | offline static feature map 대상 scan-to-map global localization |

## Build

```bash
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

## Run

```bash
roslaunch online_static_feature bag_scan_matching_static_feature.launch
roslaunch online_static_feature map_localizer.launch
```
