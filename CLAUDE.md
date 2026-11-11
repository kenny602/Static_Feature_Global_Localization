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
(실데이터 기준. Isaac Sim 시뮬레이션에서는 GT로 accuracy 평가를 추가한다 → 2.4절)
- **반복성(repeatability)**: 동일 map/scan 세션 쌍에 초기 pose perturbation
  (±0.5m, ±2°)을 N회(예: 20회) 주고 반복 실행 → 각 프레임 pose 추정치의
  표준편차/3σ.
- **시간축 jitter**: 단일 실행 내 프레임간 pose 증분의 smoothness. 동적
  객체가 지나갈 때 raw baseline은 튀고 static feature는 안정적이라는 게
  가설.
- **fitness 분포 안정성**: 정합 residual의 평균/분산.
- 보조: 프레임당 처리시간(정적 feature의 경량성 증거).

### 2.3 비교 조건: 2조건으로 확정 (2026-11-09)

- **방법 적용(ON)**: `source_mode=static`(정적 feature 스캔, feature window 3)
  + static feature 맵(surface+corner)
- **방법 미적용(OFF, baseline)**: `source_mode=accumulated`(raw 스캔, window 1)
  + raw 맵
- matcher는 `vgicp` 단일. 예전 2×2 설계(source_mode × matcher_type: ndt vs
  vgicp)는 폐기. 노은역 precision 실험과 Isaac 정확도 실험 모두 이 2조건으로
  보고한다. 시나리오별 trajectory 그림도 2조건만 그린다.
- 코드의 `features`(plane/corner 분리 발행) 모드는 남아 있지만 실험 조건에 넣지 않는다.
  등속도 예측 수정은 결과를 바꾸지 않았음.
- **feature 추출 좌표계 스위치(`extract_frame`: odom | map_init | map_live | odom_level)는
  2026-11-10 코드에서 완전히 제거**했다(재현님 결정: 스위치가 많으면 코드가 난잡). 추출은
  항상 odom 좌표계(원래 동작). 필요하면 git 이력 `1f0a821`~`af3969e`에서 복구 가능.
  제거 전 실험 결론(기록용):
  - 네 모드 간 차이 없음: Isaac GT ATE odom 0.0712 / map_init 0.0707 / map_live 0.0710 /
    odom_level 0.0711 m, 노은역 `corr_lidar_*`도 같은 수준.
  - 모드 간 fitness·`src_points_used` 차이(0.258→0.31~0.33)는 **측정 artifact**였다. odom 추출
    격자(`voxel_size` 0.5, odom 원점, floor)와 localizer의 small_gicp 축약 격자
    (`sg_downsample_resolution` 0.5)가 정확히 일치해 "추출 복셀 1개 = 축약 셀 1개"가 되고,
    격자를 2.5°만 돌려도 점이 셀을 40 % 더 차지한다. 축약을 0.4 m로 어긋나게 하면 사라진다.
    → fitness·src_points_used는 추출 방식이 다른 조건 간 비교 지표로 쓰지 말 것.
  - 중력 정렬 격자(IMU 중력으로 pillar 수직축을 세우는 것)는 future work.
- 노은역 odom 좌표계 roll/pitch 드리프트 실측: 17.6분에 최대 8.8°, 종료 시 6.1°
  (CSV `mo_roll/mo_pitch`). 이 기울기에서도 odom 추출 fitness는 유지됨.
  보정량은 반드시 센서 기준 `corr_lidar_*`로 볼 것: odom 원점 기준 `corr_dx/dy`는
  지렛대 효과로 2~3배 부풀려짐(중앙값 8.2 vs 3.9 cm, p90 36 vs 13 cm).
- **노은역은 같은 설정을 다시 돌려도 결과가 크게 달라진다**(2026-11-10, full-bag 2회씩:
  `corr_lidar` p90이 같은 설정에서 13~22 cm로 변동, 12~13분 구간에서 10~24 m 일시 이탈이
  어떤 run에는 나오고 어떤 run에는 안 나옴). 1회 실행 비교로 결론 내지 말 것 — 조건당 반복 필수.
- primitive matcher는 future work (이전 결정 유지).

### 2.4 Isaac Sim 기반 정확도(accuracy) 평가 (ROS 2 Jazzy, MAV_Localizer)

2.2의 실데이터(노은역) 실험은 GT가 없어 precision-only지만, Isaac Sim에서는
위치 GT를 얻을 수 있으므로 **정량적 정확도 평가**를 추가로 수행한다.
실행 환경은 ROS 2 Jazzy이며, 이를 위해 `online_static_feature_ws`를
ROS 2로 포팅한 `MAV_Localizer` 워크스페이스를 만들었다.

**(1) 지도 작성 (데이터 수집은 재현님이 직접 수행)**
- Isaac Sim 제공 공간 모델 안에서 라이다 센서 모델(OS2 64ch)을 장착한 로봇
  또는 지게차를 이동시키며 포인트 클라우드를 수집한다.
- 각 프레임마다 위치 정보를 `/tf`에서 기록한다.
  - `frame_id: jackal`: 전역 원점 (0, 0, 0) 좌표계 = 글로벌 맵 좌표계
  - `child_frame_id: OS2_64ch`: 라이다 센서 좌표계
- 매 프레임의 pcd(라이다 좌표계)를 `jackal`↔`OS2_64ch` transform으로 변환해
  `jackal` 좌표계로 옮기고, 이를 누적해 글로벌 맵을 구축한다.
  즉 `p_map = T_jackal_OS2(t) · p_lidar`이다. tf의 (parent=jackal, child=OS2_64ch)
  변환을 그대로 적용하면 child 좌표가 parent 좌표로 옮겨진다.
- 이 raw 맵을 `offline_static_feature_ws`로 처리해 static feature 맵을 만든다.
  raw 맵 자체는 `accumulated` baseline용 맵으로 쓴다.
- 지도를 만들 때 확인할 것:
  - **tf 시점 일치:** 각 프레임의 `header.stamp` 시점의 transform을 써야 한다.
    최신 tf를 쓰면 이동 중 프레임이 어긋난다.
  - **`jackal`이 정말 고정 좌표계인지:** 이동 중에 `jackal→OS2_64ch`
    값이 변하는지 확인한다. 값이 상수라면 센서 장착 static tf이고 `jackal`은
    로봇 몸체 좌표계다. 이 경우 world(odom)→`jackal` 변환을 추가로 곱해야
    전역 좌표가 된다.

**(2) 스캔 (localization 입력) 및 실행**
- MAV_Localizer는 외부 PC(Ubuntu 24.04, ROS 2 Jazzy)에서 동작시킨다.
- 지도는 (1)에서 만든 지도를 쓴다.
- 스캔은 TeamViewer로 원격 접속한 Isaac Sim에서 얻는다. 지도를 만든 것과
  **동일한 공간 모델**에서 동적 객체(사람) 이벤트를 활성화해, 지도 수집 당시와
  달라진 dynamic environment를 만든다. 그 상태에서 같은 로봇/지게차의 라이다
  센서 모델로 스캔한 pcd를 입력으로 쓴다.
- TeamViewer는 화면 공유일 뿐이다. 스캔 데이터는 별도로 외부 PC에 전달해야 한다.
  - **실시간:** 같은 네트워크 + 같은 `ROS_DOMAIN_ID`로 DDS를 통해 받는다.
  - **기록 후 재생:** Isaac Sim PC에서 `ros2 bag record`로 기록하고 외부 PC로 옮겨 재생한다.
  - 둘 중 어떤 방식인지 정해야 한다.
- 초기 pose(`init_x/y/z/yaw`)는 스캔 세션 첫 프레임의 GT로 줄 수 있다.
  2.2의 반복성 평가처럼 여기에 perturbation을 더할 수도 있다.

**(3) 평가: GT vs MAV_Localizer 추정 위치**
- GT는 스캔 세션의 `/tf`(`jackal`→`OS2_64ch`)에서 얻는다. 이것이 매 프레임 라이다의
  맵 좌표계 pose(`T_map_lidar`)다.
- 지도와 GT가 같은 `jackal` 좌표계이므로 Umeyama 정렬 없이 직접 비교한다.
  기존 `eval_ate.py` / `run_evo.sh`의 정책과 같다(정렬하면 실제 오차까지 흡수된다).
- 지표는 RMSE / ATE(APE), RPE를 쓴다. `MAV_Localizer/src/online_static_feature/scripts`의
  도구를 활용한다.
  - `csv_to_tum.py`: `map_localizer.csv` → 추정 궤적 TUM(`_est.tum`)
  - `run_evo.sh`: `<prefix>_gt.tum`과 `<prefix>_est.tum`으로 evo APE/RPE 계산 및 그림 생성
  - `eval_ate.py`: 프레임별 오차와 요약 통계. 단, 현재 GT 로더는 노은역 LAS
    트랙(`CameraPos/*.PosC`, UTM offset) 전용이다. Isaac Sim GT(`/tf` → TUM)
    로더를 추가해야 한다.
- 할 일: 스캔 세션 bag의 `/tf`에서 `jackal→OS2_64ch`를 뽑아 `_gt.tum`으로 저장하는
  스크립트를 만든다. 시각은 `map_localizer.csv`의 `stamp`와 같은 기준(sim time)으로 맞춘다.
- Isaac Sim은 자세 GT도 있으므로, `run_evo.sh`의 `-r trans_part`(위치만)에 더해
  회전 오차(`-r full` 또는 `angle_deg`)도 평가할 수 있다.
- 비교 조건은 2.3의 2조건(방법 적용 vs 미적용, vgicp)을 그대로 적용한다.
- **현재 상태 (2026-11-09)**: attempt_1~10 실험 완료. odom pose 비동기 버그 수정
  (3.3절) 후 결과가 최종: 위치 RMSE raw_dyn 0.0857 m vs mav_sta 0.0722 m
  (−16 %, 10 bag 중 9개 우세, 최대오차 0.52→0.39 m). 수정 전 수치(0.158/0.084,
  "47 %")는 frozen 프레임으로 부풀려진 것이므로 인용하지 말 것. 데이터·스크립트·
  결과 세대 구분은 `isaac_data/HANDOFF_PROMPT.md` 참고. 남은 일: 조건당 반복
  실행(초기 pose perturbation)으로 분산 확보.
- **이 PC(ROS1)에서도 Isaac 데이터 실험 가능 (2026-11-09 구축·검증)**:
  - 데이터: `/home/rain/data/IsaacSim_Warehouse/isaac_data/` (맵, attempt bag, GT 전부 있음).
  - `scripts/convert_isaac_mcap.sh` : Jazzy mcap → ROS1 bag(`online_scan_scenario_ros1/attempt_N.bag`,
    `/point_cloud`만). rosbags 0.9.23이 metadata version 9를 거부해 version 8로 바꾼 사본으로 읽는다.
  - `scripts/run_isaac_ros1.sh [attempts] [conds]` : 조건(raw_dyn/mav_sta…) × attempt 일괄 실행.
    `config/localization_isaac.yaml` + `config/gis_isaac_feature.yaml`(voxel 0.25 — ROS2 Isaac 실험과
    동일; 노은역은 0.5) 사용, init pose는 `attempt_N_gt/init_pose.txt`, 전용 roscore(11411).
    결과는 기본 `isaac_data/experiments/results_ros1/`(`OUT_ROOT`로 변경), 평가는 기존 `eval_isaac_ate.py --results … --out …`.
    (과거 결과 폴더 `results_ros1_{odom,map_init,map_live,odom_level,phi1,phi2}`는 제거된 스위치 시절 것)
  - 검증: attempt_1 mav_sta ATE RMSE 0.0416 m (ROS2 0.0445 m), 842/842 프레임, 10 Hz 실시간.
  - **ROS1 전체 결과 (2026-11-10, attempt 1~10, 모두 coverage 100 %, 수신=처리)**:
    raw_dyn 0.0772 m / mav_sta 0.0712 m (−8 %, 10 bag 중 9개 우세, 최대오차 0.45→0.38 m,
    >0.5 m 0.12 %→0.03 %, roll RMSE 0.094°→0.035°). ROS2와의 차이는 raw_dyn attempt_6
    (ROS2 0.157 m 일시 실패가 ROS1에서는 0.077 m)뿐. feature 격자 좌표계 A/B
    (mav_sta): odom 0.0712 / map_init 0.0707 / map_live 0.0710 / odom_level 0.0711(2026-11-10 오후)
    → **Isaac에서는 차이 없음** (bag별 차이 ≤ 1 cm, GT 정밀도 이내; 회전 RMSE 0.46~0.48°,
    roll/pitch 0.02~0.04° 전부 동일; attempt별 최고 모드는 흩어짐, odom_level이 odom보다 나은
    bag 4/10). 단 Isaac의 odom 기울기는 1~5°라 >8° 영역은 검증하지 못함. odom 좌표계 기울기 드리프트는 2~3분에
    보통 1~1.6°, attempt_2는 5.4°까지 갔지만 ATE 영향 없음.

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
   - φ 검사(점유율 > `fill_ratio_thresh`인 pillar의 법선 고도각 평균/표준편차)는 **k_min만 제외한
     나머지 복셀 전부**로 한다(2026-11-10 변경, 표본 1개면 평균 조건만). 이전에는 k_min·k_max를
     빼고 내부 복셀 ≥2일 때만 검사해서 2~3복셀 pillar는 검사 없이 k_min만 남기고 버렸다
     (fill 제거 점의 55~65 %). 변경 전후 Isaac ATE 0.0712→0.0709 m(동일), 노은역 정밀도도
     실행 간 편차 이내. offline 맵 생성 쪽(`offline_static_feature_ws`)에도 같은 규칙 반영(2026-11-10, 미커밋·
     맵 재생성 전). offline에만 있는 두 로직은 **의도적으로 유지**: (1) φ 통과 + 최상단 복셀 수평 →
     k_min만 보존(dense하게 쌓인 주차 차량 지붕 제거용), (2) 낮은 객체 필터(`use_low_object_filter`).
   - 2026-11-11 수정 3건(online·offline 모두 반영, 맵 재생성 전): (a) 점유율 분모 `k_span`→`k_span+1`(최댓값 1),
     (b) 고윳값 비(`s_thresh`)로 탈락한 pillar도 k_min이 수평이면 그 복셀만 보존(φ 탈락과 같은 규칙),
     (c) 점유율 낮은 pillar에서 수평 조건을 통과한 k_min·k_max는 pillar PCA 재검사 없이 바로 출력
     (전에는 둘을 묶어 S를 계산해 간격 ~4 m 미만이면 둘 다 버려졌다). Isaac ATE 0.0708→0.0706 m(동일),
     출력 점 +0.7 %.
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
  → **source cloud와 message_filters(ApproximateTime, slop 0.02 s)로 짝지어
  받는다** (ROS2 2026-11-06, ROS1 2026-11-09 포팅). 예전의 "버퍼에서 stamp
  검색, 실패 시 직전 pose 재사용" 방식은 부하로 pose가 늦으면 출력 자세가
  ~1 s 고정되는 버그(Isaac attempt_2)가 있어 제거. 짝 없는 프레임은 누락되며,
  종료 시 received/processed 수를 로그(ROS2)·`map_localizer_summary.txt`(ROS1)로
  남긴다. 두 수가 같으면 누락 없음.

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
