# ICRA 2026 (deadline 9/15) — Long-term Static Feature Localization

이 파일은 김재현(재현) 연구자와 Claude 간 사전 논의 내용을 요약한 것입니다.
Claude Code가 이 프로젝트 디렉토리에서 세션을 시작할 때 아래 맥락을 전제로
작업해주세요.

---

## 1. 연구 Contribution (확정)

**주 claim: Long-term global localization in dynamic environments**

핵심 문제 설정은 "map은 과거, scan은 현재"라는 시간적 간격 자체입니다.
과거에 만든 pcd map과 현재 scan 사이엔 (1) 동적 객체(그때 있던 차/사람은
지금 없고, 지금 있는 건 그때 없음), (2) 준정적 변화(주차 위치, 공사, 계절적
식생)가 누적됩니다. raw scan을 그대로 정합하면 이 시간차 노이즈가 전부 정합
오차로 들어가지만, 정적 feature 추출(corner line / planar surface)은 양쪽
모두에서 시간에 불변인 구조만 남기므로 시간 간격이 클수록 오히려 이점이
커지는 구조입니다.

**Representation-level vs. system-level 구분 (핵심 차별점)**
- 기존 lifelong localization 계열(LiLoc 등)은 system-level에서 대응: factor
  graph 내에서 map matching factor에 가중치를 주거나, error가 커지면 해당
  factor를 끄고 LIO로 지역 map을 업데이트.
- 우리는 representation-level에서 대응: registration 이전에 static feature
  extraction으로 오차 원인 자체를 제거. 지도를 계속 업데이트할 필요가 없는
  정보만 지도로 가져간다는 게 핵심 selling point.
- 미지 영역으로 지도를 확장하는 exploration 문제는 스코프 밖(별도 연구
  영역)임을 서론에서 명확히 경계 지을 것.

**Dynamic object removal은 목적이 아니라 수단**
학습 기반 동적 객체 제거(DRR-LIO, DOF-LIO 등)와 제거 성능으로 경쟁하는
방향이 아님. 우리 방법은 포인트 분포 기하 정보만으로 동작 → 학습 데이터나
GPU 인프라가 불필요하다는 경량성이 contribution.

**Odometry 적용 가능성 (지도교수 논의 결과, 2026-07-30)**
- 교수님이 "정적 feature가 odometry(short-term, relative pose)에도 적용
  가능하지 않냐"는 논의를 제기했으나, 최종적으로 **ICRA 방향은 그대로
  long-term global localization 유지**로 확정됨.
- odometry 적용 가능성은 논문 discussion/future work에 한 단락으로만 언급.
- 이유: (1) 컨퍼런스 논문이므로 contribution을 무겁게 가져가지 않고 명확하게,
  (2) odometry 방향으로 가면 DRR-LIO/DOF-LIO 등 동적 객체 제거 전문
  연구들과 정면 경쟁해야 함, (3) 시간 제약상 실험 프레임워크를 이원화할
  여유 없음.

**Primitive matcher (LOAM식 correspondence)는 future work로 확정**
- 원래 8월 중 구현 예정이었던 EdgePrimitive/PlanePrimitive 기반 correspondence
  matcher는 지도교수 논의 결과 **future work로 공식 확정**됨.
- ICRA 실험 범위는 "correspondence를 찾을 필요가 없는 NDT 혹은 GICP 계열"로
  scan-to-map matching만 구현하는 것으로 축소 확정. → 이게 이미 구현된
  `map_localizer_node`와 정확히 일치 (아래 3절 참고).

---

## 2. 실험 설계 (확정)

### 2.1 데이터: private 노은역 데이터 우선, NCLT는 후순위 확장

- **주 실험**: 보유 중인 "이전 노은역" 데이터(map용/scan용 세션 이미 분리됨)
  + 앞으로 받을 "새 노은역" 데이터(7월 예정, offline raw pcd + online bag).
  이것만으로 실험이 완결됨 (새 데이터 지연되어도 논문 성립).
- **NCLT는 명칭과 달리 같은 날 수집이 아님** (정정된 사실 — 15개월, 27세션,
  2주 간격 수집, 계절/날씨/공사 변화 포함). 시간이 남으면 8월 하순 확장
  용도로 고려: (a) GT가 있어 accuracy(ATE) 보조 지표 확보 가능, (b) HDL-Loc
  등 published baseline과 직접 비교 가능. 단, 센서가 Velodyne HDL-32E라
  Ouster 대비 파라미터 재튜닝 필요.
- **Published baseline (LiLoc, HDL-Loc, polex 등)**: 지금은 하지 않음 (내부
  ablation 우선). 8월 하순 여유 있으면 재검토. bag에 IMU 토픽 있음 확인됨
  → LiLoc을 L+I reference 계층으로 노은역 데이터에서 돌리는 게 polex보다
  현실적 (polex는 NCLT 로더 밀착 + 2D pose + pole 밀도 의존이라 이식 부담 큼).

### 2.2 평가 metric: 정확도(accuracy) 아님, 정밀도(precision)만

GT 없음 → precision-only 평가로 확정.
- **반복성(repeatability)**: 동일 map/scan 세션 쌍에 초기 pose perturbation
  (±0.5m, ±2°)을 N회(예: 20회) 주고 반복 실행 → 각 프레임 pose 추정치의
  표준편차/3σ.
- **시간축 jitter**: 단일 실행 내 프레임간 pose 증분의 smoothness. 동적
  객체가 지나갈 때 raw baseline은 튀고 static feature는 안정적이라는 게
  가설.
- **fitness 분포 안정성**: 정합 residual의 평균/분산.
- 보조: 프레임당 처리시간(정적 feature의 경량성 증거).

### 2.3 Ablation 매트릭스 (2×2, 확장 가능)

- **축 1 (source_mode)**: `accumulated`(raw dense window, baseline) vs.
  `features`(정적 feature, 제안)
- **축 2 (matcher_type)**: `ndt` vs. `gicp`/`vgicp`(small_gicp) — backend
  무관하게 정적 feature가 유리함을 보이기 위해 2개 backend로 검증
- primitive matcher가 future work로 빠졌으므로 이 2×2가 최종 실험 범위.
  (예전엔 여기에 `primitive` 축 추가를 계획했으나 확정 스코프에서 제외됨)

---

## 3. 코드베이스 구조 (파악 완료)

Repo 2개:
- `kenny602/offline_static_feature_ws` — offline map 생성용
- `kenny602/online_static_feature_ws` — online 파이프라인 (아래 상세)

### 3.1 노드 3개, 실행 흐름: accumulator → online feature → (신규) localizer

**[노드1] `scan_matching_accumulator_node`**
- raw LiDAR(`/ouster/points`) → preprocess(range/z 크롭) → submap 대상 정합
  (small_gicp, VGICP 기본) → `T_odom_lidar` 추정 → 발행: `scan_matching_pose`,
  `dense_window_cloud`.
- **실제 운용 launch(`bag_scan_matching_static_feature.launch`) 값은
  window_frames=1, submap_frames=1** (소스 기본값 4/6이 아님 — launch가
  override). 즉 scan-to-scan 수준 정합. 원래 재현님이 말한 "raw 연속 2개
  frame 누적"이 맞는 설명이었음(직전 1프레임 target + 현재 1프레임 source).
- registration_type 기본값: VGICP.

**[노드2] `online_static_feature_node`**
- `dense_window_cloud` + `scan_matching_pose` 동기화 수신 → (옵션)
  transform_with_pose로 map 좌표계 변환 → 추가 누적(feature_window_frames,
  실제 운용값 1) → `buildVoxelAndPillarMap` → `classifyAllPillars` → 색상
  기반 plane/corner 분리 발행.
- 발행 토픽(실제 값, ns 포함): `scan_plane_features`, `scan_corner_features`,
  `accumulated_cloud` (모두 `/online_static_feature_node/` 네임스페이스).
- 색상 규약: 흰색=PLANAR, 파란색=CORNER(pillar 전체), 빨간색=VOXEL_CORNER
  (벽 pillar 내부의 개별 edge point). 노드는 흰색=plane, 파랑+빨강=corner로
  묶어 발행.

**[노드3] `pcd_window_feature_batch_node`**
- 저장된 PCD 시퀀스를 오프라인 배치로 처리 (offline map 생성/디버깅용).

### 3.2 정적 feature 추출 (`classifyAllPillars`, `pillar_pca.cpp`) — 동적 제거 6단계

voxel(0.1~0.2m) → 같은 (i,j) voxel을 pillar로 묶어 판정. 동적/비정적 판정은
**단일 필터가 아니라 6개 층**으로 구성 (각각 독립 `use_*` 플래그 + 카운터):

1. voxel min points — voxel 내 포인트 수 미달 시 제거
2. voxel PCA S-filter (`use_voxel_filter`, `voxel_s_thresh`) — planarity/
   scatter 지표로 blob 판정
3. span filter (`use_span_filter`, `H_min_voxel`) — pillar 수직 span 부족
   시 전체 제거
4. fill-ratio/sandwich/φ-통계 — 점유율 기반 벽/바닥·천장/동적 판정
5. classifyPillar 자체 — pillar 전체 PCA S값 + normal 일관성(σ_θ, `n_bar`
   기준 각 θ의 평균/표준편차)로 PLANAR/CORNER/DYNAMIC 판정
6. 2-pass 공간 필터 — unknown-space filter(고립 pillar 제거) + floor-height
   filter(주변 local 바닥 median 대비 튀는 voxel 제거)

실제 GIS config에서는 이 중 일부만 켜서 사용 중(span filter + classifyPillar
+ φ 위주). 이 6단계 구조는 향후 세분화 ablation(어떤 필터가 정밀도에
실제로 기여하는지) 재료로 활용 가능하나 우선순위는 낮음.

**참고 (지난 오류 정정 이력)**: 이전 논의에서 "emit"을 잘못된 3단 구조로
설명한 적 있음(잘못된 설명, 코드 미확인 상태의 추측이었음). 실제로는 코드에
`EmitKind{PLANAR, CORNER, VOXEL_CORNER}` enum이 존재하며 위 6단계 필터를
통과한 뒤 최종 출력 클라우드에 넣는 단계를 가리킴. `n_bar`는 z성분 기반
평면 분리가 아니라 normal 일관성 판정의 기준 벡터임.

### 3.3 신규 구현: `map_localizer_node.cpp` (완료, 빌드 검증 전)

위치: `online_static_feature_ws/src/online_static_feature/src/map_localizer_node.cpp`
(+ `launch/map_localizer.launch`, `CMakeLists.txt` 수정 완료)

**역할**: offline static feature map(pcd) 대상 scan-to-map localization.
`T_odom_lidar`(accumulator가 이미 odom frame으로 발행)에 `T_map_odom`
보정치만 추정하는 구조. `T_map_lidar = T_map_odom × T_odom_lidar`.

**핵심 설계**:
- 초기 pose는 `init_x/y/z/yaw` 파라미터로 수동 지정 (LSMCL의 initialize
  모듈이 있다는 가정을 대신).
- 매 프레임: 이전 `T_map_odom`을 initial guess로 소스 클라우드를 offline
  맵에 정합 → odometry 예측 대비 보정량이 임계(기본 2m/10°) 넘으면 reject.
- **파라미터 스위치 2개로 2×2 ablation 매트릭스 전체 커버**:
  - `matcher_type`: `ndt`(PCL NDT) / `gicp`,`vgicp`,`icp`,`plane_icp`
    (small_gicp, 맵은 시작 시 1회만 전처리 캐시)
  - `source_mode`: `features`(plane+corner 합침) / `accumulated`(raw
    dense window, baseline)
- CSV 로그(`log_csv_path`)에 프레임별 pose, 보정량(corr_dx..dyaw), fitness,
  수렴 여부, reject 여부, 정합 시간 기록 → precision 평가(2.2절)용 후처리
  입력으로 바로 사용 가능.

**빌드 전 확인 필요 사항**:
1. small_gicp의 `preprocess_points`/`align(target, source, tree, init,
   setting)` 시그니처가 로컬 small_gicp 버전과 일치하는지 (버전별 오버로드
   차이 가능)
2. CMakeLists에 PCL `registration` 컴포넌트 추가함(NDT 용) — 링크 에러
   확인 필요
3. **아직 실제 catkin_make 빌드/실행 검증 전** — 다음 세션에서 최우선
   작업.

**토픽명 (실제값으로 수정 완료)**:
- plane: `/online_static_feature_node/scan_plane_features`
- corner: `/online_static_feature_node/scan_corner_features`
- accumulated: `/online_static_feature_node/accumulated_cloud`
- odom pose: `/scan_matching_accumulator_node/scan_matching_pose`

**첫 검증 절차 (self-localization)**: map을 생성한 바로 그 bag으로
localizer를 돌려 T_map_odom이 identity 근처에서 안정적인지 확인 → 통과 후
다른 세션 bag(시간차 있는 실제 시나리오)으로 교차 검증.

---

## 4. 관련 연구 포지셔닝 (서베이 논문과 공유)

관련연구 3그룹으로 구분해 서론/related work 작성:

1. **Map-update 계열 lifelong localization** (LiLoc, LL-Localizer 등):
   환경 변화에 지도를 계속 업데이트. 한계: 업데이트 비용, 계절별 지도
   분리 등 운영 부담. → 우리는 "업데이트 불필요한 정보만 지도화."
2. **Landmark 기반 long-term localization** (polex/Schaefer et al., RAS
   2021): 시간 불변 구조를 남긴다는 철학은 공유하나 특정 랜드마크(pole)
   의존 → 랜드마크 희소 환경에서 실패 가능. → 우리는 일반적 corner/plane
   구조로 일반화. (NCLT 15개월 장기 평가 프로토콜은 우리 precision 평가
   설계의 근거로 인용 가능)
3. **Dynamic 환경 odometry / 동적 객체 제거** (DRR-LIO, DOF-LIO, 예림님
   labmate 연구): 상대 위치 추정 문제, 학습 기반 제거는 데이터/GPU 인프라
   필요. → 우리는 절대 위치(global localization) 문제이며 기하 분포만으로
   동작.

Schaefer et al.(pole, RAS 2021)이 가장 근접한 선행 비교 대상.

병행 중인 한국어 서베이 논문(안아름, 이유철 공저): 위 3그룹 분류 체계와
Localization_Feature_Extraction_Survey repo 내용을 그대로 활용 가능.

---

## 5. 일정 (8주, 7/30 기준)

- **~7월 말**: `map_localizer_node` 빌드·self-localization 검증
- **8월 1~3주**: perturbation 반복 실행 자동화 + precision 후처리 스크립트
  → 2×2 ablation 본 실험 (보유 데이터)
- **8월 4주**: 새 노은역 데이터 도착 시 교체 실험 / 여유 있으면 LiLoc
  reference 실행 or NCLT 확장 검토
- **9월 1~2주**: 논문 작성 (related work는 서베이 원고 재활용)
- **9월 2주**: 마무리, 제출

## 6. 다음 세션 시작 시 우선 작업

1. `catkin_make -DCMAKE_BUILD_TYPE=Release` 빌드 → 에러 있으면 small_gicp
   시그니처/PCL 링크부터 확인
2. self-localization 검증 (map 생성 bag = localizer 입력 bag)
3. 통과하면 perturbation 반복 실행 스크립트 작성 → 2×2 precision 실험 착수
