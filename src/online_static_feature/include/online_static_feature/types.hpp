#pragma once

#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <limits>

// ──────────────────────────────────────────────
//  Voxel 좌표 (i, j, k)
// ──────────────────────────────────────────────
struct VoxelKey
{
    int64_t i, j, k;
    bool operator==(const VoxelKey &o) const
    {
        return i == o.i && j == o.j && k == o.k;
    }
};

struct VoxelKeyHash
{
    // XOR + 대형 소수 곱셈 기반 해시 (modulo 제거)
    // - 기존: (((k * P) % M + j) * P) % M + i  → modulo 2회 (느림)
    // - 현재: 각 축을 독립 소수로 곱한 뒤 XOR  → modulo 0회
    //   (PCL OctoMap 등에서 쓰이는 표준 3D voxel 해시 패턴)
    size_t operator()(const VoxelKey &v) const
    {
        size_t h = static_cast<size_t>(v.i) * 73856093u;
        h ^= static_cast<size_t>(v.j) * 19349663u;
        h ^= static_cast<size_t>(v.k) * 83492791u;
        return h;
    }
};

// ──────────────────────────────────────────────
//  Pillar 좌표 (i, j)
// ──────────────────────────────────────────────
struct PillarKey
{
    int64_t i, j;
    bool operator==(const PillarKey &o) const { return i == o.i && j == o.j; }
};

struct PillarKeyHash
{
    // XOR + 대형 소수 곱셈 기반 해시 (modulo 제거)
    // - 기존: (j * P) % M + i  → modulo 1회
    // - 현재: i, j 독립 소수 곱 후 XOR  → modulo 0회
    size_t operator()(const PillarKey &p) const
    {
        size_t h = static_cast<size_t>(p.i) * 73856093u;
        h ^= static_cast<size_t>(p.j) * 19349663u;
        return h;
    }
};

// ──────────────────────────────────────────────
//  Voxel 하나의 포인트 묶음 (per-voxel PCA용)
// ──────────────────────────────────────────────
struct VoxelPoints
{
    int64_t                      k_index; // voxel의 k 인덱스 (높이 방향)
    std::vector<Eigen::Vector3f> points;

    // Voxel PCA는 span filter와 normal consistency에서 반복 참조된다.
    // mutable cache로 const 경로에서도 한 번만 계산해 재사용한다.
    mutable bool pca_computed = false;
    mutable bool pca_valid    = false;
    mutable double lambda1 = 0.0, lambda2 = 0.0, lambda3 = 0.0;
    mutable Eigen::Vector3d normal = Eigen::Vector3d::Zero();
};

// ──────────────────────────────────────────────
//  Pillar 데이터
//  - point_count : pillar 전체 포인트 수
//  - voxels      : voxel_map이 소유한 voxel을 가리키는 포인터 목록
// ──────────────────────────────────────────────
struct PillarData
{
    size_t point_count = 0;
    std::vector<const VoxelPoints *> voxels;
};

// ──────────────────────────────────────────────
//  분류 레이블
//  PLANAR       : 흰색  — 천장/바닥/벽 (mu 낮고 sigma 낮음)
//  STRONG_CORNER: 빨간색 — 벽-벽/벽-바닥 교선 (mu 낮고 sigma 높음)
//  CORNER       : 파란색 — 일반 corner (mu 높음)
//  DYNAMIC      : 제거
// ──────────────────────────────────────────────
enum class Label : uint8_t
{
    DYNAMIC       = 0,
    PLANAR        = 1,
    CORNER        = 2,
    // STRONG_CORNER = 3,
    SKIP          = 4
};

// ──────────────────────────────────────────────
//  Pillar 전체 PCA 결과 (Idea 1)
// ──────────────────────────────────────────────
struct PCAResult
{
    double lambda1 = 0, lambda2 = 0, lambda3 = 0;
    double S = 1.0;   // sphericity  = λ3/λ1
    double L = 0.0;   // linearity   = (λ1-λ2)/λ1
    double P = 0.0;   // planarity   = (λ2-λ3)/λ1
    Eigen::Vector3d n_min = Eigen::Vector3d::Zero(); // 최소 eigenvalue의 eigenvector (surface normal)
    bool   valid = false;
};

// ──────────────────────────────────────────────
//  Normal 일관성 결과 (Idea 2)
// ──────────────────────────────────────────────
struct NormalConsistency
{
    double mu_theta    = 999.0;  // 평균 각도 편차 [rad]
    double sigma_theta = 999.0;  // 표준편차 [rad]
    std::vector<double> theta_v;    // per-voxel theta값 (voxels와 같은 순서로 저장)
    std::vector<uint8_t> theta_valid; // 해당 voxel의 theta/PCA 성공 여부
    bool   valid       = false;
};

// ──────────────────────────────────────────────
//  파라미터
// ──────────────────────────────────────────────
struct ClassifyParams
{
    // ---- 전처리: 다운샘플링 -------------------------
    // 0 = 다운샘플링 안 함
    // 1 = VoxelGrid 다운샘플링 적용
    int    use_downsample      = 1;
    float  downsample_size     = 0.05f;  // [m] 다운샘플링 복셀 크기

    // ---- 전처리: RANSAC 바닥 보정 -------------------
    // z축을 실제 수직 방향과 정렬
    // 0 = 사용 안 함
    // 1 = 전체 포인트 중 z 하위 N% 로 RANSAC 바닥 평면 추정
    int    use_floor_align     = 1;
    float  floor_ransac_thresh = 0.05f;  // [m] RANSAC inlier 거리
    float  floor_z_percentile  = 10.0f; // [%] 바닥 후보 포인트 비율
    int    floor_ransac_iters  = 1000;  // RANSAC 반복 횟수

    // ---- Voxel 단위 dynamic 필터링 ------------------
    // per-voxel PCA로 S = λ3/λ1 계산
    // S >= voxel_s_thresh 인 voxel은 동적 객체로 판단 → pillar에서 제외
    // 0 = 사용 안 함, 1 = 사용
    int    use_voxel_filter    = 0;
    double voxel_s_thresh      = 0.05;  // 이 이상이면 blob → dynamic voxel

    // ---- 천장 존재 여부 판정 (RANSAC) -----------------
    // z값 상위 N% 포인트로 RANSAC → 후보 평면의 normal n_ceil 추정
    // |n_ceil · ẑ| >= ceiling_parallel_thresh 이고 inlier 비율이 충분하면
    //   → ceiling_exists = true (바닥과 평행한 천장 존재)
    // 그렇지 않으면 → ceiling_exists = false (주차장 등 천장 없음)
    float  ceiling_z_percentile    = 10.0f; // [%] 천장 후보 포인트 비율 (z 상위)
    float  ceiling_ransac_thresh   = 0.05f; // [m] RANSAC inlier 거리
    int    ceiling_ransac_iters    = 1000;  // RANSAC 반복 횟수
    double ceiling_parallel_thresh = 0.95;  // |n_ceil · ẑ| 기준
    double ceiling_inlier_ratio_min = 30.0; // [%] inlier 비율 최소 기준

    // ---- [실험용] 천장 없는 환경 시뮬레이션 ------------
    // floor alignment 완료 후 z 상위 N% 포인트를 제거하여
    // 천장이 없는 것처럼 만듦. 주차장 등 실외 테스트 용.
    // 0 = 사용 안 함 (원본 유지)
    // 1 = 사용 (z 상위 N% 제거)
    int    simulate_no_ceiling        = 0;
    float  ceiling_cut_percentile     = 20.0f; // [%] 제거할 z 상위 비율

    // ---- k_span 필터 + Sandwich 필터 ----------------
    // k_span < H_min_voxel 인 pillar → 전체 제거 (노이즈/낮은 물체)
    //
    // 천장 있음 (ceiling_exists=true):
    //   fill_ratio <= fill_ratio_thresh 인 pillar에 sandwich 필터 적용
    //   bot voxel: α_bot >= horiz_normal_thresh → 보존 (바닥)
    //   top voxel: α_top >= horiz_normal_thresh → 보존 (천장)
    //   나머지 voxel: 모두 제거
    //
    // 천장 없음 (ceiling_exists=false):
    //   fill_ratio 조건 사용하지 않음. 모든 pillar 대상.
    //   k_min 제외한 voxel들의 φ_v = arcsin(α_v) 로 mean/std 계산
    //     φ_mean < phi_mean_thresh AND φ_std < phi_std_thresh:
    //       → 벽/기둥 pillar → 전체 보존
    //     그 외:
    //       → 동적/혼합 pillar → k_min 만 α 로 체크 후 나머지 제거
    int    use_span_filter       = 0;
    int    H_min_voxel           = 10;   // 최소 k_span (voxel 단위)
    double fill_ratio_thresh     = 0.5;  // 이 이하면 sandwich 구조로 판단 (천장 있음)
    double horiz_normal_thresh   = 0.8;  // α = |n̂ · ẑ| 이 이상이면 수평면 → 보존
                                         // 권장 시작값: 0.8 (약 37° 이내)

    // ---- local floor 대비 수평 voxel 높이 필터 -------
    // 단일 voxel pillar 또는 span/fill 필터에서 바닥/천장 후보로 살아남은
    // 수평 voxel이 주변 bottom voxel median보다 일정 높이 이상이면 제거한다.
    int    use_floor_height_filter = 0;
    int    floor_height_radius     = 3;   // 주변 탐색 반경 (pillar grid 단위)
    double floor_height_max        = 0.5; // local floor 대비 허용 높이 [m]

    // ---- [천장 없음 분기] φ 통계 기반 벽 pillar 판정 ---
    // φ_v = arcsin(α_v) ∈ [0, π/2]
    //   φ ≈ 0   → 수직면 (벽)
    //   φ ≈ π/2 → 수평면
    // 벽 pillar: φ_mean 작고 φ_std 작음 (일관된 수직)
    // 동적/혼합: φ_mean 변동 또는 φ_std 큼
    // use_phi_weight: 1이면 f(N)=1-exp(-3N/n_ref)로 phi mean/std 가중
    int    use_phi_weight       = 0;
    double phi_mean_thresh       = 0.35;  // [rad] ≈ 20° : 이 이하 → 수직 우세
    double phi_std_thresh        = 0.26;  // [rad] ≈ 15° : 이 이하 → 일관됨

    // --- Voxel / Pillar 구성 ---
    float  voxel_size      = 0.1f;  // [m]
    int    n_voxel_min     = 4;      // voxel 내 최소 포인트 수
    int    n_pillar_min    = 10;     // pillar 내 최소 포인트 수

    // --- Idea 1: Sphericity ---
    double s_thresh        = 0.05;   // 이 이상 → DYNAMIC

    // --- Idea 2: Normal 일관성 ---
    // use_weight: 0 = f(N) 없이 단순 평균
    //             1 = f(N) = 1-exp(-3N/N_ref) 가중 평균/분산
    int    use_weight      = 0;
    double n_ref           = 20.0;   // f(N_ref) ≈ 0.95가 되는 포인트 수

    // PLANAR 조건: mu <= mu_thresh AND sigma <= sigma_thresh
    // STRONG_CORNER 조건: mu <= mu_thresh AND sigma > sigma_thresh
    // CORNER 조건: mu > mu_thresh
    double mu_thresh       = 0.35;   // [rad] ≈ 20° : 이 이하 → PLANAR
    double sigma_thresh    = 0.18;   // [rad] ≈ 20° : 이 이상 → DYNAMIC

    // --- Per-voxel corner 탐지 (PLANAR pillar 내부) ---
    // 벽-바닥 / 벽-천장 사이 부분 edge 판별을 위함
    // PLANAR로 분류된 pillar 내 각 voxel의 각도차이가
    // median * corner_multiplier 이상이면 해당 voxel을 CORNER로 분류
    // 0 = 사용 안함, 1 = 사용
    int    use_voxel_corner  = 1;
    double corner_multiplier = 5.0;  // median 대비 배수

    // --- Corner 다운샘플링 (Centroid) ---
    // CORNER로 분류된 voxel의 포인트를 voxel centroid 1점으로 대표
    //   ĉ_v = (1/N_v) Σ p_i
    // ImMesh에서 edge mesh를 일관되게 생성하기 위한 전처리
    // 0 = 사용 안 함 (모든 corner 포인트 출력)
    // 1 = 사용 (voxel당 centroid 1점 출력)
    int    use_corner_downsample = 0;

    // --- PLANAR/CORNER 분류 사용 여부 ---
    // 0 = 동적 객체 필터링(classifyPillar의 DYNAMIC/SKIP 판정)까지만 수행하고,
    //     생존 pillar의 포인트를 분류 없이 그대로 출력한다. voxel corner 탐색과
    //     PLANAR/CORNER 색 결정을 건너뛰므로 그만큼 연산이 줄어든다.
    //     use_corner_downsample=0 기준으로 출력 점 집합은 1일 때와 동일하다.
    // 1 = 기존 동작 (PLANAR/CORNER 분류 + voxel corner 탐색)
    // 클래스 라벨을 쓰는 registration은 future work이므로, scan-to-map 정합만
    // 하는 현재 단계에서는 0으로 두어 불필요한 연산을 피할 수 있다.
    int    use_feature_classification = 1;

    // --- classifyAllPillars OpenMP 스레드 수 ---
    // pillar는 서로 독립이라 병렬 실행이 안전하다.
    // 같은 머신에서 accumulator(num_threads)와 map_localizer(sg_num_threads)가
    // 동시에 도므로 코어 수를 다 쓰면 경합한다. 1이면 직렬 실행.
    int    classify_num_threads = 4;

    // --- buildVoxelAndPillarMap Step1 OpenMP 스레드 수 ---
    // unordered_map은 동시 삽입이 불가능하므로, 스레드별로 map을 따로 두고
    // hash(key) % T == tid 인 키만 담당한다. 키가 겹치지 않아 병합 시
    // 충돌이 없고, node handle splice로 옮겨서 VoxelPoints 주소를 보존한다
    // (PillarData.voxels가 그 주소를 가리키므로 반드시 유지되어야 한다).
    // 1이면 기존 단일 map 직렬 경로를 그대로 탄다.
    int    voxel_num_threads = 1;

    // --- Planar(surface) feature voxel-filter 다운샘플 ---
    // scan/map matching에서 KD-Tree를 구성할 점 집합을 줄이기 위한 후처리
    // (LSMCL 수식 17 관점). corner는 손대지 않고 planar 점만 대상.
    // 큰 bbox + 작은 leaf 시 pcl::VoxelGrid 정수 인덱스 오버플로가 나므로
    // per-window(merge 전) surface 클라우드에 적용한다.
    // 0 = 사용 안 함, 1 = 사용 (voxel당 centroid 1점)
    int    use_planar_downsample = 1;
    float  planar_leaf           = 0.05f; // [m] surface 다운샘플 복셀 크기

    // --- Corner distance-spacing 다운샘플 (LSMCL Eq.3 d_c) ---
    // corner 점을 anchor끼리 항상 corner_spacing 이상 떨어지도록 greedy 선별.
    // 출력 _corner_edge.pcd: 선별 anchor(파랑/빨강) + 인접 anchor 초록 세그먼트.
    // 0 = 사용 안 함, 1 = 사용
    int    use_corner_edge       = 1;
    float  corner_spacing        = 0.1f;   // [m] anchor greedy distance-spacing 간격
};
