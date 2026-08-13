#include "pillar_pca.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <unordered_set>

// ─────────────────────────────────────────────────────────────────────────
//  공통 헬퍼
// 병목 분석용 누적기. thread_local이라 이후 OpenMP 병렬화에도 안전하다.
// classifyAllPillars 진입 시 0으로 초기화하고, 종료 시 ClassifyTiming에 옮긴다.
thread_local double g_eigen_voxel_ms = 0.0;
thread_local long   g_eigen_voxel_cnt = 0;

// ─────────────────────────────────────────────────────────────────────────
static bool eigenDecompose(const std::vector<Eigen::Vector3f> &pts,
                            double &l1, double &l2, double &l3,
                            Eigen::Vector3d &evec_min)
{
    const auto t_eig0 = std::chrono::steady_clock::now();
    struct Acc {
        std::chrono::steady_clock::time_point t;
        ~Acc() {
            g_eigen_voxel_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t).count();
            ++g_eigen_voxel_cnt;
        }
    } acc{t_eig0};

    const int N = static_cast<int>(pts.size());
    if (N < 3) return false;

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto &p : pts) mean += p.cast<double>();
    mean /= static_cast<double>(N);

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto &p : pts)
    { Eigen::Vector3d d = p.cast<double>()-mean; cov += d*d.transpose(); }
    cov /= static_cast<double>(N);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success) return false;

    l1 = solver.eigenvalues()(2);
    l2 = solver.eigenvalues()(1);
    l3 = solver.eigenvalues()(0);
    evec_min = solver.eigenvectors().col(0).normalized();
    return true;
}

static bool ensureVoxelPCA(const VoxelPoints &voxel)
{
    if (voxel.pca_computed)
        return voxel.pca_valid;

    voxel.pca_computed = true;
    voxel.pca_valid = eigenDecompose(voxel.points,
                                     voxel.lambda1,
                                     voxel.lambda2,
                                     voxel.lambda3,
                                     voxel.normal);
    return voxel.pca_valid;
}

static double voxelCountWeight(const VoxelPoints &voxel, double n_ref)
{
    if (n_ref <= 1e-9) return 1.0;
    const double Nv = static_cast<double>(voxel.points.size());
    return 1.0 - std::exp(-3.0 * Nv / n_ref);
}

static bool eigenDecomposePillar(const PillarData &pillar,
                                 double &l1, double &l2, double &l3,
                                 Eigen::Vector3d &evec_min)
{
    const int N = static_cast<int>(pillar.point_count);
    if (N < 3) return false;

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto *vox : pillar.voxels)
    {
        if (!vox) continue;
        for (const auto &p : vox->points)
            mean += p.cast<double>();
    }
    mean /= static_cast<double>(N);

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto *vox : pillar.voxels)
    {
        if (!vox) continue;
        for (const auto &p : vox->points)
        {
            Eigen::Vector3d d = p.cast<double>() - mean;
            cov += d * d.transpose();
        }
    }
    cov /= static_cast<double>(N);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success) return false;

    l1 = solver.eigenvalues()(2);
    l2 = solver.eigenvalues()(1);
    l3 = solver.eigenvalues()(0);
    evec_min = solver.eigenvectors().col(0).normalized();
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
//  Voxel 단위 PCA
// ─────────────────────────────────────────────────────────────────────────
PCAResult computeVoxelPCA(const VoxelPoints &voxel)
{
    PCAResult result;
    if (!ensureVoxelPCA(voxel)) return result;
    result.lambda1 = voxel.lambda1;
    result.lambda2 = voxel.lambda2;
    result.lambda3 = voxel.lambda3;
    if (result.lambda1 < 1e-10) return result;
    result.S = result.lambda3 / result.lambda1;
    result.L = (result.lambda1 - result.lambda2) / result.lambda1;
    result.P = (result.lambda2 - result.lambda3) / result.lambda1;
    result.n_min = voxel.normal; // surface normal 저장
    result.valid = true;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────
//  Pillar 전체 PCA
// ─────────────────────────────────────────────────────────────────────────
PCAResult computePillarPCA(const PillarData &pillar)
{
    PCAResult result;
    Eigen::Vector3d evec_min;
    if (!eigenDecomposePillar(pillar,
                              result.lambda1, result.lambda2, result.lambda3,
                              evec_min)) return result;
    if (result.lambda1 < 1e-10) return result;
    result.S = result.lambda3 / result.lambda1;
    result.L = (result.lambda1 - result.lambda2) / result.lambda1;
    result.P = (result.lambda2 - result.lambda3) / result.lambda1;
    result.valid = true;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────
//  Normal 일관성
// ─────────────────────────────────────────────────────────────────────────
NormalConsistency computeNormalConsistency(const PillarData &pillar,
                                            int use_weight, double n_ref,
                                            bool valid_only)
{
    NormalConsistency nc;
    const int num_voxels = static_cast<int>(pillar.voxels.size());

    struct VN { Eigen::Vector3d n; double w; int vox_idx; };
    std::vector<VN> vnormals;
    vnormals.reserve(num_voxels);

    for (int vi = 0; vi < num_voxels; ++vi)
    {
        const auto *vox = pillar.voxels[vi];
        if (!vox || !ensureVoxelPCA(*vox)) continue;
        double w = 1.0;
        if (use_weight==1)
        {
            w = voxelCountWeight(*vox, n_ref);
        }
        vnormals.push_back({vox->normal, w, vi});
    }

    if (vnormals.size() < 2) return nc;

    const Eigen::Vector3d &n0 = vnormals[0].n;
    for (auto &vn : vnormals)
        if (vn.n.dot(n0) < 0.0) vn.n = -vn.n;

    Eigen::Vector3d n_bar = Eigen::Vector3d::Zero();
    for (const auto &vn : vnormals) n_bar += vn.w * vn.n;
    if (n_bar.norm() < 1e-10) return nc;
    n_bar.normalize();

    // valid_only: 분류를 쓰지 않으면 classifyPillar가 nc.valid만 본다.
    // θv/μ_θ/σ_θ와 theta_v/theta_valid는 voxel corner 분기에서만 읽히는데
    // 그 분기를 건너뛰므로, pillar당 벡터 할당 4회와 voxel당 acos + 분산 2차
    // 순회를 생략한다. sum_w 가드는 전체 경로와 동일하게 유지해 valid 판정이
    // 달라지지 않게 한다.
    if (valid_only)
    {
        double sum_w = 0.0;
        for (const auto &vn : vnormals) sum_w += vn.w;
        if (sum_w <= 1e-12) return nc;
        nc.valid = true;
        return nc;
    }

    // theta_v/theta_valid: 모든 voxel에 대해 크기 고정
    // → nc.theta_v.size() == pillar.voxels.size() 항상 보장
    // PCA 실패 voxel은 theta_valid=0으로 표시해 median 계산에서 제외한다.
    std::vector<double> thetas(num_voxels, 0.0);
    std::vector<uint8_t> theta_valid(num_voxels, 0);
    double sum_w = 0.0;
    double sum_theta = 0.0;

    for (const auto &vn : vnormals)
    {
        double c = std::fabs(vn.n.dot(n_bar));
        c = std::min(1.0, std::max(0.0, c));
        double theta = std::acos(c);
        thetas[vn.vox_idx] = theta;
        theta_valid[vn.vox_idx] = 1;
        sum_w += vn.w;
        sum_theta += vn.w * theta;
    }

    if (sum_w <= 1e-12) return nc;
    nc.mu_theta = sum_theta / sum_w;
    double var=0;
    for (const auto &vn : vnormals)
    {
        const double d = thetas[vn.vox_idx] - nc.mu_theta;
        var += vn.w * d*d;
    }
    nc.sigma_theta = std::sqrt(var / sum_w);
    nc.theta_v = thetas;  // 전체 voxel 크기로 저장
    nc.theta_valid = theta_valid;
    nc.valid = true;
    return nc;
}

// ─────────────────────────────────────────────────────────────────────────
//  분류
// ─────────────────────────────────────────────────────────────────────────
Label classifyPillar(const PCAResult &pca, const NormalConsistency &nc,
                     const ClassifyParams &params)
{
    if (!pca.valid) return Label::SKIP;
    if (pca.S >= params.s_thresh) return Label::DYNAMIC;
    if (!nc.valid) return Label::DYNAMIC;
    // 동적 판정은 위에서 끝났다. 분류를 쓰지 않으면 μ_θ/σ_θ가 계산되지 않았으므로
    // (기본값 999.0) 비교하지 않고 PLANAR로 반환한다. PLANAR/CORNER 둘 다
    // 생존하므로 출력 점 집합에는 영향이 없다.
    if (!params.use_feature_classification) return Label::PLANAR;
    if (nc.mu_theta <= params.mu_thresh &&
        nc.sigma_theta <= params.sigma_thresh)
        return Label::PLANAR;
    return Label::CORNER;
}

// ─────────────────────────────────────────────────────────────────────────
//  Voxel Centroid
//
//  voxel 내 모든 포인트의 산술 평균 좌표를 계산한다.
//  반환값은 합성 좌표 (원본 포인트 아님).
// ─────────────────────────────────────────────────────────────────────────
static Eigen::Vector3f voxelCentroid(const std::vector<Eigen::Vector3f> &pts)
{
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto &p : pts) c += p;
    return c / static_cast<float>(pts.size());
}


// ─────────────────────────────────────────────────────────────────────────
//  전체 pillar 분류
//
//  필터링 순서:
//  [1] voxel PCA filter  : S >= voxel_s_thresh 인 voxel 제거
//  [2] k_span filter     : k_span < H_min_voxel → pillar 전체 제거
//  [3] fill_ratio filter : occupied_ratio <= fill_ratio_thresh 이면
//                          k_min/k_max 근방만 유지, 사이 제거
//  [4] normal 일관성     : PLANAR vs CORNER 분류
// ─────────────────────────────────────────────────────────────────────────
pcl::PointCloud<pcl::PointXYZRGB>::Ptr classifyAllPillars(
    const std::unordered_map<PillarKey, PillarData, PillarKeyHash> &pillar_map,
    const ClassifyParams &params,
    float floor_z_ref,
    int &out_planar_count,
    int &out_corner_count,
    int &out_voxel_dynamic_count,
    int &out_pillar_min_count,
    int &out_pillar_dynamic_count,
    int &out_span_dynamic_count,
    int &out_fill_dynamic_count,
    int &out_voxel_corner_count,
    int &out_wall_pillar_count,
    int &out_corner_ds_removed,
    ClassifyTiming *timing)
{
    using ProfClock = std::chrono::steady_clock;
    using ProfMs = std::chrono::duration<double, std::milli>;
    double prof_math_ms = 0.0;
    double prof_emit_ms = 0.0;
    double prof_phi_ms = 0.0;
    long prof_pillars = 0;
    g_eigen_voxel_ms = 0.0;
    g_eigen_voxel_cnt = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    (void)floor_z_ref;  // height filter는 local neighbor median 기준을 사용한다.
    const bool needs_height_filter = params.use_floor_height_filter;
    const bool needs_post_filter = needs_height_filter;

    // output points 재할당 최소화를 위한 reserve
    //   대략 입력 포인트 수 정도를 최대로 가정 (동적 객체 제거로 실제로는 더 적음)
    size_t total_input_points = 0;
    for (const auto &kv : pillar_map)
        total_input_points += kv.second.point_count;
    output->points.reserve(total_input_points);

    out_planar_count        = 0;
    out_corner_count        = 0;
    out_voxel_dynamic_count = 0;
    out_pillar_min_count    = 0;
    out_pillar_dynamic_count= 0;
    out_span_dynamic_count  = 0;
    out_fill_dynamic_count  = 0;
    out_voxel_corner_count  = 0;
    out_wall_pillar_count   = 0;
    out_corner_ds_removed   = 0;

    enum class EmitKind : uint8_t { PLANAR, CORNER, VOXEL_CORNER };
    struct EmittedPoint
    {
        PillarKey key;
        int64_t voxel_k;
        pcl::PointXYZRGB point;
        EmitKind kind;
        int ds_removed;
    };

    // ── 병합 대상 (병렬 영역 바깥) ────────────────────────────────────────
    std::vector<EmittedPoint> emitted_points;
    if (needs_post_filter)
        emitted_points.reserve(total_input_points);
    std::unordered_map<PillarKey, PillarData, PillarKeyHash> survived_map;
    if (needs_post_filter)
        survived_map.reserve(pillar_map.size());
    std::unordered_map<PillarKey, std::unordered_set<int64_t>, PillarKeyHash>
        height_filter_voxel_keys;

    // 루프 이후 2-pass 필터에서도 쓰이므로 병렬 영역 밖에 둔다. 순수 함수다.
    auto meanZOfVoxel = [](const VoxelPoints *vox) -> double {
        if (!vox || vox->points.empty()) return 0.0;
        double sum_z = 0.0;
        for (const auto &pt : vox->points)
            sum_z += static_cast<double>(pt.z());
        return sum_z / static_cast<double>(vox->points.size());
    };

    // ── OpenMP를 이용한 병렬 실행 준비 ───────────────────────────────────────
    // pillar는 서로 독립이다. voxel은 (i,j,k), pillar는 (i,j)라 한 voxel이
    // 정확히 한 pillar에만 속하므로, VoxelPoints의 mutable PCA 캐시를 두
    // 스레드가 동시에 건드릴 일이 없다.
    // omp for는 random access가 필요해서 map을 벡터로 평탄화한다.
    std::vector<const std::pair<const PillarKey, PillarData> *> pillar_list;
    pillar_list.reserve(pillar_map.size());
    for (const auto &kv : pillar_map)
        pillar_list.push_back(&kv);

    // 병렬 영역 안에서는 같은 이름의 지역 변수가 아래 변수들을 가리므로
    // (shadowing) 병합할 때는 포인터로 접근한다. 이 덕분에 루프 본문은
    // 한 줄도 바뀌지 않는다.
    auto *g_output   = &output;
    auto *g_emitted  = &emitted_points;
    auto *g_survived = &survived_map;
    auto *g_hfvk     = &height_filter_voxel_keys;
    int *g_planar = &out_planar_count,   *g_corner = &out_corner_count;
    int *g_voxdyn = &out_voxel_dynamic_count, *g_pmin = &out_pillar_min_count;
    int *g_pdyn = &out_pillar_dynamic_count,  *g_span = &out_span_dynamic_count;
    int *g_fill = &out_fill_dynamic_count,    *g_vcorner = &out_voxel_corner_count;
    int *g_wall = &out_wall_pillar_count,     *g_cds = &out_corner_ds_removed;
    double g_math_ms = 0.0, g_emit_ms = 0.0, g_phi_ms = 0.0;
    double g_eig_ms = 0.0;
    long g_pillars = 0, g_eig_cnt = 0;

    const int classify_threads = std::max(1, params.classify_num_threads);
    const size_t reserve_per_thread =
        total_input_points / static_cast<size_t>(classify_threads) + 1;

    #pragma omp parallel num_threads(classify_threads)
    {
    // ── 스레드별 상태 (바깥 변수를 같은 이름으로 가림) ────────────────
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    output->points.reserve(reserve_per_thread);
    std::vector<EmittedPoint> emitted_points;
    if (needs_post_filter)
        emitted_points.reserve(reserve_per_thread);
    std::unordered_map<PillarKey, PillarData, PillarKeyHash> survived_map;
    std::unordered_map<PillarKey, std::unordered_set<int64_t>, PillarKeyHash>
        height_filter_voxel_keys;
    int out_planar_count = 0, out_corner_count = 0, out_voxel_dynamic_count = 0;
    int out_pillar_min_count = 0, out_pillar_dynamic_count = 0;
    int out_span_dynamic_count = 0, out_fill_dynamic_count = 0;
    int out_voxel_corner_count = 0, out_wall_pillar_count = 0;
    int out_corner_ds_removed = 0;
    double prof_math_ms = 0.0, prof_emit_ms = 0.0, prof_phi_ms = 0.0;
    long prof_pillars = 0;
    g_eigen_voxel_ms = 0.0;    // thread_local
    g_eigen_voxel_cnt = 0;

    auto addEmittedPoint = [&](const PillarKey &key,
                               int64_t voxel_k,
                               const Eigen::Vector3f &pt,
                               uint8_t r, uint8_t g, uint8_t b,
                               EmitKind kind,
                               int ds_removed = 0) {
        pcl::PointXYZRGB p;
        p.x = pt.x(); p.y = pt.y(); p.z = pt.z();
        p.r = r; p.g = g; p.b = b;
        if (needs_post_filter)
        {
            emitted_points.push_back({key, voxel_k, p, kind, ds_removed});
            return;
        }

        output->push_back(p);
        if (kind == EmitKind::PLANAR)
        {
            out_planar_count += 1;
        }
        else if (kind == EmitKind::CORNER)
        {
            out_corner_count += 1;
            out_corner_ds_removed += ds_removed;
        }
        else
        {
            out_corner_count += 1;
            out_voxel_corner_count += 1;
            out_corner_ds_removed += ds_removed;
        }
    };

    auto storeSurvivedVoxel = [&](const PillarKey &key, const VoxelPoints *vox) {
        if (!needs_post_filter || !vox) return;
        PillarData pd;
        pd.point_count = vox->points.size();
        pd.voxels.push_back(vox);
        survived_map[key] = std::move(pd);
    };

    auto storeSurvivedPillar = [&](const PillarKey &key, const PillarData &pillar) {
        if (!needs_post_filter) return;
        survived_map[key] = pillar;
    };

    // ── 공통 헬퍼: voxel 포인트를 PLANAR(흰색) 출력 후보로 저장 ───────────
    auto emitPlanarVoxel = [&](const PillarKey &key,
                               const VoxelPoints *vox,
                               bool use_height_filter = false) {
        if (!vox) return;
        storeSurvivedVoxel(key, vox);
        if (needs_height_filter && use_height_filter)
            height_filter_voxel_keys[key].insert(vox->k_index);
        for (const auto &pt : vox->points)
            addEmittedPoint(key, vox->k_index, pt, 255, 255, 255,
                            EmitKind::PLANAR);
    };
 
    // ── 공통 헬퍼: voxel normal과 z축 내적 (α) ──────────────────────────────
    const Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
    auto alphaOfVoxel = [&z_axis](const VoxelPoints *vox) -> double {
        if (!vox) return 0.0;
        if (!ensureVoxelPCA(*vox))
            return 0.0;
        return std::fabs(vox->normal.dot(z_axis));
    };

    auto kMinOfPillar = [](const PillarData &pillar) -> int64_t {
        int64_t k_min = std::numeric_limits<int64_t>::max();
        for (const auto *vox : pillar.voxels)
            if (vox) k_min = std::min(k_min, vox->k_index);
        return k_min;
    };

    #pragma omp for schedule(dynamic, 32) nowait
    for (size_t pillar_idx = 0; pillar_idx < pillar_list.size(); ++pillar_idx)
    {
        const auto &kv = *pillar_list[pillar_idx];
        const PillarData &pillar_orig = kv.second;

        // ── [1] Voxel PCA filter ─────────────────────────────────────────
        PillarData pillar_after_voxel;
        const PillarData *ptr = &pillar_orig;
 
        if (params.use_voxel_filter)
        {
            pillar_after_voxel.voxels.reserve(pillar_orig.voxels.size());
 
            for (const auto *vox : pillar_orig.voxels)
            {
                if (!vox || static_cast<int>(vox->points.size()) < params.n_voxel_min)
                    continue;
                PCAResult vr = computeVoxelPCA(*vox);
                if (vr.valid && vr.S >= params.voxel_s_thresh)
                { out_voxel_dynamic_count += static_cast<int>(vox->points.size()); continue; }
                pillar_after_voxel.point_count += vox->points.size();
                pillar_after_voxel.voxels.push_back(vox);
            }
            ptr = &pillar_after_voxel;
        }
 
        // ── [2-3] Per-pillar 통합 필터 ───────────────────────────────────
        PillarData pillar_after_span;
        bool is_wall_pillar = true;
 
        if (params.use_span_filter && !ptr->voxels.empty())
        {
            // k_min/k_max + bot/top 포인터 (1회 순회)
            int64_t k_min = std::numeric_limits<int64_t>::max();
            int64_t k_max = std::numeric_limits<int64_t>::lowest();
            const VoxelPoints *vox_bot = nullptr;
            const VoxelPoints *vox_top = nullptr;
            for (const auto *vox : ptr->voxels)
            {
                if (!vox) continue;
                if (vox->k_index < k_min) { k_min = vox->k_index; vox_bot = vox; }
                if (vox->k_index > k_max) { k_max = vox->k_index; vox_top = vox; }
            }
            const int64_t k_span = k_max - k_min;
 
            // k_span < H_min_voxel → pillar 전체 제거
            // 단, voxel 1개짜리 pillar는 아래 바닥 전용 처리로 넘긴다.
            if (ptr->voxels.size() != 1 &&
                k_span < static_cast<int64_t>(params.H_min_voxel))
            {
                out_span_dynamic_count += static_cast<int>(ptr->point_count);
                continue;
            }
 
            const int occupied = static_cast<int>(ptr->voxels.size());
            const double fill_ratio = (k_span > 0)
                ? static_cast<double>(occupied) / static_cast<double>(k_span)
                : 1.0;
 
            // per-pillar α_top (천장 유무 판정에 사용)
            const double alpha_top = alphaOfVoxel(vox_top);
            const bool top_is_horiz = (alpha_top >= params.horiz_normal_thresh);
 
            // ────────────────────────────────────────────────────────────
            // Case 1: voxel 1개 — 바닥 전용 처리
            // ────────────────────────────────────────────────────────────
            if (ptr->voxels.size() == 1)
            {
                const double alpha_bot = alphaOfVoxel(vox_bot);
                if (alpha_bot >= params.horiz_normal_thresh && vox_bot)
                    emitPlanarVoxel(kv.first, vox_bot, true);
                else
                    out_fill_dynamic_count += static_cast<int>(ptr->point_count);
                continue;
            }
 
            // ────────────────────────────────────────────────────────────
            // Case 2: voxel 2개 이상
            // ────────────────────────────────────────────────────────────
 
            // ── Case 2A: fill_ratio 높음 → φ 통계로 벽/동적 판정 ─────────
            if (fill_ratio > params.fill_ratio_thresh)
            {
                // k_min & k_max 제외한 중간 voxel로 φ mean/std 계산
                const auto t_phi0 = ProfClock::now();
                struct PhiSample { double phi; double w; };
                std::vector<PhiSample> phi_list;
                phi_list.reserve(ptr->voxels.size());
                for (const auto *vox : ptr->voxels)
                {
                    if (!vox || vox == vox_bot || vox == vox_top) continue;
                    const double a = alphaOfVoxel(vox);
                    const double a_clamped = std::min(1.0, std::max(0.0, a));
                    const double w = (params.use_phi_weight == 1)
                        ? voxelCountWeight(*vox, params.n_ref)
                        : 1.0;
                    phi_list.push_back({std::asin(a_clamped), w});
                }
 
                bool is_wall_phi = false;
                if (phi_list.size() >= 2)
                {
                    double sum_w = 0.0;
                    double sum_phi = 0.0;
                    for (const auto &sample : phi_list)
                    {
                        sum_w += sample.w;
                        sum_phi += sample.w * sample.phi;
                    }
                    const double denom = (sum_w > 1e-12)
                        ? sum_w
                        : static_cast<double>(phi_list.size());
                    const double phi_mean = sum_phi / denom;
                    double var = 0.0;
                    for (const auto &sample : phi_list)
                    {
                        const double d = sample.phi - phi_mean;
                        var += sample.w * d*d;
                    }
                    const double phi_std = std::sqrt(var / denom);
 
                    is_wall_phi = (phi_mean < params.phi_mean_thresh) &&
                                  (phi_std  < params.phi_std_thresh);
                }
                prof_phi_ms += ProfMs(ProfClock::now() - t_phi0).count();
 
                if (is_wall_phi)
                {
                    // Outdoor ETRI scans have many stable wall-like pillars whose
                    // top voxel can still look horizontal. Preserve the full pillar
                    // and let Stage 4 classify it instead of applying the old
                    // indoor box/desk k_min-only heuristic.
                }
                else
                {
                    // φ 불일관 → 동적/혼합 → k_min만 보존
                    is_wall_pillar = false;
                    const double alpha_bot = alphaOfVoxel(vox_bot);
                    const bool bot_ok = (alpha_bot >= params.horiz_normal_thresh) && vox_bot;
                    for (const auto *vox : ptr->voxels)
                    {
                        if (!vox) continue;
                        if (vox == vox_bot && bot_ok) continue;
                        out_fill_dynamic_count += static_cast<int>(vox->points.size());
                    }
                    if (bot_ok) emitPlanarVoxel(kv.first, vox_bot, true);
                    continue;
                }
            }
            // ── Case 2B: fill_ratio 낮음 → per-pillar 천장 판정 ─────────
            else
            {
                is_wall_pillar = false;
 
                const double alpha_bot = alphaOfVoxel(vox_bot);
                const bool keep_bot = (alpha_bot >= params.horiz_normal_thresh) && vox_bot;
                bool keep_top = top_is_horiz && vox_top;  // per-pillar 천장 판정
                if (needs_height_filter && keep_bot)
                    height_filter_voxel_keys[kv.first].insert(vox_bot->k_index);
                if (needs_height_filter && keep_top)
                    height_filter_voxel_keys[kv.first].insert(vox_top->k_index);
 
                pillar_after_span.voxels.reserve(2);
 
                for (const auto *vox : ptr->voxels)
                {
                    if (!vox) continue;
                    const bool is_bot = (vox == vox_bot);
                    const bool is_top = (vox == vox_top);
                    const bool keep = (is_bot && keep_bot) || (is_top && keep_top);
                    if (keep)
                    {
                        pillar_after_span.point_count += vox->points.size();
                        pillar_after_span.voxels.push_back(vox);
                    }
                    else
                    {
                        out_fill_dynamic_count += static_cast<int>(vox->points.size());
                    }
                }
 
                // bot/top만 남은 경우 PLANAR 직접 출력 후 Stage 4 건너뜀
                if (pillar_after_span.voxels.size() <= 1)
                {
                    if (keep_bot) emitPlanarVoxel(kv.first, vox_bot, true);
                    if (keep_top && !keep_bot) emitPlanarVoxel(kv.first, vox_top, true);
                    continue;
                }
 
                ptr = &pillar_after_span;
            }
        }

        // ── [4] Pillar 최소 포인트 체크 ──────────────────────────────────
        if (static_cast<int>(ptr->point_count) < params.n_pillar_min)
        {
            out_pillar_min_count += static_cast<int>(ptr->point_count);
            continue;
        }

        // ── [4] PCA + Normal 일관성 + 분류 ───────────────────────────────
        const auto t_math0 = ProfClock::now();
        PCAResult pca = computePillarPCA(*ptr);

        NormalConsistency nc;
        if (pca.valid && pca.S < params.s_thresh)
            nc = computeNormalConsistency(*ptr, params.use_weight, params.n_ref,
                                          !params.use_feature_classification);

        Label lbl = classifyPillar(pca, nc, params);
        prof_math_ms += ProfMs(ProfClock::now() - t_math0).count();
        ++prof_pillars;

        if (lbl == Label::DYNAMIC || lbl == Label::SKIP)
        { out_pillar_dynamic_count += static_cast<int>(ptr->point_count); continue; }

        storeSurvivedPillar(kv.first, *ptr);

        // ── 분류를 쓰지 않는 경우 여기서 종료 ────────────────────────────
        // 동적 객체 제거는 위 [1]~[4]에서 이미 끝났다. 아래 색상 결정과
        // voxel corner 탐색(theta 정렬/median)은 PLANAR/CORNER 라벨을 만들기
        // 위한 것이고, scan-to-map 정합은 두 라벨을 다시 합쳐 쓰므로 불필요하다.
        // use_corner_downsample=0이면 아래 분기도 결국 pillar의 모든 포인트를
        // 그대로 출력하므로, 여기서 끊어도 출력 점 집합은 동일하다.
        if (!params.use_feature_classification)
        {
            const auto t_emit0 = ProfClock::now();
            for (const auto *vox : ptr->voxels)
            {
                if (!vox) continue;
                for (const auto &pt : vox->points)
                    addEmittedPoint(kv.first, vox->k_index, pt, 255, 255, 255,
                                    EmitKind::PLANAR);
            }
            prof_emit_ms += ProfMs(ProfClock::now() - t_emit0).count();
            continue;
        }

        // ── 색상 결정 ─────────────────────────────────────────────────────
        uint8_t r=0, g=0, b=0;
        if (lbl == Label::CORNER)
        {
            if (params.use_corner_downsample)
            {
                // Per-voxel centroid: 각 voxel의 포인트 평균 좌표 1점 출력
                //   ĉ_v = (1/N_v) Σ p_i
                for (const auto *vox : ptr->voxels)
                {
                    if (!vox) continue;
                    Eigen::Vector3f c = voxelCentroid(vox->points);
                    addEmittedPoint(kv.first, vox->k_index, c, 0, 0, 255,
                                    EmitKind::CORNER,
                                    static_cast<int>(vox->points.size()) - 1);
                }
                continue;  // 일괄 출력 건너뜀
            }
            r=0; g=0; b=255;
        }
        else // PLANAR -> voxel corner 결정
        {
            if (params.use_voxel_corner &&
                is_wall_pillar &&
                nc.valid &&
                nc.theta_v.size() == ptr->voxels.size() &&
                nc.theta_valid.size() == ptr->voxels.size())
            {
                out_wall_pillar_count++;  // 탐지 대상 벽 pillar 수

                std::vector<double> sorted_t;
                sorted_t.reserve(nc.theta_v.size());
                for (size_t vi = 0; vi < nc.theta_v.size(); ++vi)
                {
                    if (nc.theta_valid[vi])
                        sorted_t.push_back(nc.theta_v[vi]);
                }
                if (!sorted_t.empty())
                {
                    std::sort(sorted_t.begin(), sorted_t.end());
                    double median_theta = sorted_t[sorted_t.size()/2];
                    double thresh_theta = median_theta * params.corner_multiplier;

                    for (size_t vi = 0; vi < ptr->voxels.size(); ++vi)
                    {
                        if (nc.theta_valid[vi] && nc.theta_v[vi] > thresh_theta)
                        {
                            const auto *vox = ptr->voxels[vi];
                            if (!vox) continue;
                            if (params.use_corner_downsample)
                            {
                                Eigen::Vector3f c = voxelCentroid(vox->points);
                                addEmittedPoint(
                                    kv.first, vox->k_index, c,
                                    255, 0, 0, EmitKind::VOXEL_CORNER,
                                    static_cast<int>(vox->points.size()) - 1);
                            }
                            else
                            {
                                for (const auto &pt : vox->points)
                                    addEmittedPoint(kv.first, vox->k_index,
                                                    pt, 255, 0, 0,
                                                    EmitKind::VOXEL_CORNER);
                            }
                        }
                        else
                        {
                            const auto *vox = ptr->voxels[vi];
                            if (!vox) continue;
                            for (const auto &pt : vox->points)
                                addEmittedPoint(kv.first, vox->k_index,
                                                pt, 255, 255, 255,
                                                EmitKind::PLANAR);
                        }
                    }
                    continue;
                }
            }

            // 단순 PLANAR (voxel corner 없음) → 흰색
            r=255; g=255; b=255;
        }

        // ── 색상 일괄 적용 (CORNER pillar, 단순 PLANAR) ──────────────────
        for (const auto *vox : ptr->voxels)
        {
            if (!vox) continue;
            for (const auto &pt : vox->points)
                addEmittedPoint(kv.first, vox->k_index, pt, r, g, b,
                                lbl == Label::CORNER ? EmitKind::CORNER
                                                     : EmitKind::PLANAR);
        }
    }   // omp for 끝

    // ── 스레드별 결과 병합 ────────────────────────────────────────────
    // 출력 포인트 순서는 스레드 스케줄에 따라 달라지지만, registration은
    // 순서에 무관하다. 점 집합 자체는 직렬 실행과 동일하다.
    #pragma omp critical
    {
        (*g_output)->insert((*g_output)->end(),
                            output->begin(), output->end());
        if (needs_post_filter)
        {
            g_emitted->insert(g_emitted->end(),
                              emitted_points.begin(), emitted_points.end());
            for (auto &e : survived_map)
                (*g_survived)[e.first] = std::move(e.second);
            for (auto &e : height_filter_voxel_keys)
                (*g_hfvk)[e.first].insert(e.second.begin(), e.second.end());
        }
        *g_planar  += out_planar_count;
        *g_corner  += out_corner_count;
        *g_voxdyn  += out_voxel_dynamic_count;
        *g_pmin    += out_pillar_min_count;
        *g_pdyn    += out_pillar_dynamic_count;
        *g_span    += out_span_dynamic_count;
        *g_fill    += out_fill_dynamic_count;
        *g_vcorner += out_voxel_corner_count;
        *g_wall    += out_wall_pillar_count;
        *g_cds     += out_corner_ds_removed;
        g_math_ms  += prof_math_ms;
        g_emit_ms  += prof_emit_ms;
        g_phi_ms   += prof_phi_ms;
        g_pillars  += prof_pillars;
        g_eig_ms   += g_eigen_voxel_ms;    // thread_local, 스레드별 값
        g_eig_cnt  += g_eigen_voxel_cnt;
    }
    }   // omp parallel 끝

    // 스레드별 누적값을 함수 스코프 변수로 되돌린다.
    prof_math_ms = g_math_ms;
    prof_emit_ms = g_emit_ms;
    prof_phi_ms  = g_phi_ms;
    prof_pillars = g_pillars;

    // 최종 특징 분류까지 살아남은 pillar를 대상으로 주변 pillar의 바닥면 높이 평균의 median과 비교
    // 단, 적용대상은 아래와 같음
    // ---- 1. voxel 1개짜리 pillar에서 살아남은 voxel (바닥면으로 판정)
    // ---- 2. fill_ratio가 낮은 pillar에서 살아남은 k_max voxel (천장으로 판정)
    std::unordered_map<PillarKey, std::unordered_set<int64_t>, PillarKeyHash>
        height_remove_voxel_keys;
    if (needs_height_filter)
    {
        auto bottomVoxelOfPillar = [](const PillarData &pillar)
            -> const VoxelPoints * {
            const VoxelPoints *bottom = nullptr;
            int64_t k_min = std::numeric_limits<int64_t>::max();
            for (const auto *vox : pillar.voxels)
            {
                if (!vox) continue;
                if (vox->k_index < k_min)
                {
                    k_min = vox->k_index;
                    bottom = vox;
                }
            }
            return bottom;
        };

        auto localBottomMedianZ = [&](const PillarKey &key,
                                      double &median_z) -> bool {
            std::vector<double> bottom_zs;
            const int r = params.floor_height_radius;
            const int r2 = r * r;

            for (int di = -r; di <= r; ++di)
            {
                for (int dj = -r; dj <= r; ++dj)
                {
                    if (di == 0 && dj == 0) continue;
                    if (di * di + dj * dj > r2) continue;

                    PillarKey nb{key.i + di, key.j + dj};
                    auto nb_it = survived_map.find(nb);
                    if (nb_it == survived_map.end() ||
                        nb_it->second.voxels.empty())
                        continue;

                    const VoxelPoints *bottom =
                        bottomVoxelOfPillar(nb_it->second);
                    if (!bottom) continue;
                    bottom_zs.push_back(meanZOfVoxel(bottom));
                }
            }

            if (bottom_zs.empty()) return false;
            std::sort(bottom_zs.begin(), bottom_zs.end());
            const size_t mid = bottom_zs.size() / 2;
            if (bottom_zs.size() % 2 == 0)
                median_z = 0.5 * (bottom_zs[mid - 1] + bottom_zs[mid]);
            else
                median_z = bottom_zs[mid];
            return true;
        };

        for (const auto &kv : height_filter_voxel_keys)
        {
            auto pillar_it = survived_map.find(kv.first);
            if (pillar_it == survived_map.end()) continue;

            double local_floor_z = 0.0;
            const bool has_local_floor =
                localBottomMedianZ(kv.first, local_floor_z);

            for (const auto *vox : pillar_it->second.voxels)
            {
                if (!vox || kv.second.find(vox->k_index) == kv.second.end()) continue;
                if (!has_local_floor ||
                    meanZOfVoxel(vox) > local_floor_z + params.floor_height_max)
                {
                    height_remove_voxel_keys[kv.first].insert(vox->k_index);
                }
            }
        }
    }

    // ── 최종 output 생성 및 생존 feature 카운트 ──────────────────────────
    for (const auto &ep : emitted_points)
    {
        auto hr_it = height_remove_voxel_keys.find(ep.key);
        if (hr_it != height_remove_voxel_keys.end() &&
            hr_it->second.find(ep.voxel_k) != hr_it->second.end())
        {
            out_fill_dynamic_count += 1 + ep.ds_removed;
            continue;
        }

        output->push_back(ep.point);
        if (ep.kind == EmitKind::PLANAR)
        {
            out_planar_count += 1;
        }
        else if (ep.kind == EmitKind::CORNER)
        {
            out_corner_count += 1;
            out_corner_ds_removed += ep.ds_removed;
        }
        else
        {
            out_corner_count += 1;
            out_voxel_corner_count += 1;
            out_corner_ds_removed += ep.ds_removed;
        }
    }

    output->width=static_cast<uint32_t>(output->points.size());
    output->height=1; output->is_dense=true;

    if (timing)
    {
        timing->pillar_math_ms = prof_math_ms;
        timing->emit_ms = prof_emit_ms;
        timing->phi_ms = prof_phi_ms;
        // 병렬 실행 시 스레드별 시간의 합이라 벽시계 시간보다 크다.
        // 스레드 수로 나누면 대략의 벽시계 기여분이 된다.
        timing->voxel_pca_ms = g_eig_ms;
        timing->voxel_pca_cnt = g_eig_cnt;
        timing->pillars_seen = prof_pillars;
    }
    return output;
}
