#pragma once

#include "types.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// ── RANSAC 바닥 평면 보정 ────────────────────
// z값 하위 N% 포인트로 RANSAC → 바닥 normal 추출
// 바닥 normal을 [0,0,1]로 맞추는 3D 회전 행렬 R 계산 후 전체 포인트에 적용
// RANSAC floor inlier들의 회전 후 평균 z를 0으로 맞추기 위해 z 평행이동
// out_floor_z: 보정 후 바닥 기준 z (항상 0, 필터 기준용)
// 반환: 회전 적용된 새 포인트 클라우드
pcl::PointCloud<pcl::PointXYZ>::Ptr correctFloorAlignment(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    const ClassifyParams &params,
    Eigen::Matrix3f &out_R,
    float &out_floor_z);   // 적용된 회전 행렬, 회전 후 바닥 z (로그/필터용)

// ── [실험용] z 상위 N% 포인트 제거 ────────────
// floor alignment 이후 호출. 천장 없는 환경을 시뮬레이션하기 위해
// z값 상위 cut_percentile% 포인트를 잘라낸 새 cloud 반환.
pcl::PointCloud<pcl::PointXYZ>::Ptr cropPointsAboveZ(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    float cut_percentile);

// ── 포인트 클라우드 수평 회전 ─────────────────
// 모든 포인트를 z축 기준으로 -theta_rad 만큼 회전한 새 클라우드 반환.
// 회전된 좌표계가 벽 방향과 정렬되도록 하기 위함.
pcl::PointCloud<pcl::PointXYZ>::Ptr rotateCloudXY(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    double theta_rad);
 
// ── Voxel Map + Pillar Map 구성 ───────────────
void buildVoxelAndPillarMap(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    const ClassifyParams &params,
    std::unordered_map<VoxelKey,  VoxelPoints, VoxelKeyHash>  &voxel_map,
    std::unordered_map<PillarKey, PillarData,  PillarKeyHash> &pillar_map,
    int &out_voxel_min_removed);
