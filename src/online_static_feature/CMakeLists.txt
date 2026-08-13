cmake_minimum_required(VERSION 3.0.2)
project(online_static_feature)

add_compile_options(-std=c++17 -O3)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_filters
  pcl_conversions
  pcl_ros
  roscpp
  nav_msgs
  sensor_msgs
  std_msgs
  tf2_ros
  visualization_msgs
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io registration)
find_package(OpenMP)

set(SMALL_GICP_ROOT "/home/rain/small_gicp_ws/small_gicp" CACHE PATH
  "Path to the small_gicp source tree")
set(SMALL_GICP_INCLUDE_DIR "${SMALL_GICP_ROOT}/include")

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs message_filters nav_msgs pcl_conversions pcl_ros roscpp sensor_msgs std_msgs tf2_ros visualization_msgs
)

include_directories(
  include/online_static_feature
  ${SMALL_GICP_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(online_static_feature_node
  src/online_static_feature_node.cpp
  src/voxel_map.cpp
  src/pillar_pca.cpp
)

target_link_libraries(online_static_feature_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  # classifyAllPillars의 pillar 루프 병렬화에 필요.
  # 이게 없으면 #pragma omp가 조용히 무시되어 직렬로 빌드된다.
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

add_executable(scan_matching_accumulator_node
  src/scan_matching_accumulator_node.cpp
  ${SMALL_GICP_ROOT}/src/small_gicp/registration/registration.cpp
  ${SMALL_GICP_ROOT}/src/small_gicp/registration/registration_helper.cpp
)

target_link_libraries(scan_matching_accumulator_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

add_executable(map_localizer_node
  src/map_localizer_node.cpp
  ${SMALL_GICP_ROOT}/src/small_gicp/registration/registration.cpp
  ${SMALL_GICP_ROOT}/src/small_gicp/registration/registration_helper.cpp
)

target_link_libraries(map_localizer_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

add_executable(pcd_window_feature_batch_node
  src/pcd_window_feature_batch_node.cpp
  src/voxel_map.cpp
  src/pillar_pca.cpp
)

target_link_libraries(pcd_window_feature_batch_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  # pillar_pca.cpp를 공유하므로 동일하게 필요하다.
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

install(TARGETS
  online_static_feature_node
  scan_matching_accumulator_node
  map_localizer_node
  pcd_window_feature_batch_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
