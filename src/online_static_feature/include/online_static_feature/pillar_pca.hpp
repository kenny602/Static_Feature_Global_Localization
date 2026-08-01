#pragma once

#include "types.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// S >= voxel_s_thresh 이면 blob(dynamic) → 해당 voxel 제외
PCAResult computeVoxelPCA(const VoxelPoints &voxel);

// ── Idea 1: Pillar 전체 PCA ───────────────────────────────────────────────
PCAResult computePillarPCA(const PillarData &pillar);

// ── Idea 2: Per-voxel normal 일관성 ──────────────────────────────────────
//
//  각 voxel마다 local PCA → normal nv (λ_min eigenvector)
//  신뢰도 가중치 f(Nv) = 1 - exp(-3Nv / N_ref), f(N_ref) ≈ 0.95
//  sign 정렬 후 가중평균 normal n̄ 계산
//  θv = arccos(|nv · n̄|)
//  μ_θ = weighted mean(θv),  σ_θ = weighted std(θv)
NormalConsistency computeNormalConsistency(const PillarData &pillar,
                                           int use_weight,
                                           double n_ref);

// ── 최종 분류 ─────────────────────────────────────────────────────────────
//
//  1차: S < s_thresh          → Static 후보
//       S >= s_thresh         → DYNAMIC
//
//  2차: σ_θ > sigma_thresh    → DYNAMIC (normal 방향 불일치)
//       μ_θ <= mu_thresh      → PLANAR  (normal 방향 수렴, 평면)
//       μ_θ >  mu_thresh      → CORNER  (static이지만 elevated mean angle)
Label classifyPillar(const PCAResult       &pca,
                     const NormalConsistency &nc,
                     const ClassifyParams   &params);

// ── 전체 pillar 순회 → 색상 포인트 클라우드 생성 ─────────────────────────
// per-pillar 천장 판정 + 통합 필터링
pcl::PointCloud<pcl::PointXYZRGB>::Ptr classifyAllPillars(
    const std::unordered_map<PillarKey, PillarData, PillarKeyHash> &pillar_map,
    const ClassifyParams &params,
    float floor_z_ref,
    int &out_planar_count,
    int &out_corner_count,
    int &out_voxel_dynamic_count,   // voxel PCA 필터 제거
    int &out_pillar_min_count,      // n_pillar_min 미만 제거
    int &out_pillar_dynamic_count,  // pillar PCA/normal 유효성 제거
    int &out_span_dynamic_count,    // k_span 기반 제거 (너무 낮은 pillar)
    int &out_fill_dynamic_count,    // sandwich pillar 내 bot/top 외 voxel 제거
    int &out_voxel_corner_count,    // PLANAR 벽 pillar 내 빨간색 voxel corner 포인트 수
    int &out_wall_pillar_count,     // per-voxel corner 탐지 대상 벽 pillar 수
    int &out_corner_ds_removed);    // FPS 다운샘플링으로 제거된 corner 포인트 수
