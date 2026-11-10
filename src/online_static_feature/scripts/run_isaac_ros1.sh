#!/usr/bin/env bash
# Isaac Sim 창고 데이터 2조건 실험 (ROS1 / 이 PC).
# ROS2 쪽 isaac_data/experiments/run_isaac_localization.sh 와 같은 조건 정의를 쓴다.
#
#   ./run_isaac_ros1.sh [attempts] [conditions]
#     attempts   : 예) "1 2 3"            (기본 1..10)
#     conditions : raw_dyn mav_sta        (기본; raw_sta mav_dyn 도 가능)
#
#   raw_* : source_mode=accumulated, feature_window_frames=1   (방법 OFF)
#   mav_* : source_mode=static,      feature_window_frames=3   (방법 ON)
#   *_dyn : prior map = gt_map/dynamic/$MAP_NAME/gt_map.pcd
#   *_sta : prior map = gt_map/static/$MAP_NAME/gt_map_{surface,corner}.pcd
#
# env:
#   ISAAC_DATA    기본 /home/rain/data/IsaacSim_Warehouse/isaac_data
#   BAG_DIR       기본 $ISAAC_DATA/online_scan_scenario_ros1  (convert_isaac_mcap.sh 산출물)
#   OUT_ROOT      기본 $ISAAC_DATA/experiments/results_ros1
#   MAP_NAME      기본 dynamic_environment_1
#   MATCHER       기본 vgicp      RATE 기본 1.0 (bag 재생 배속)      DRAIN 기본 10 (bag 끝난 뒤 대기 s)
#   ROS_PORT      기본 11411 (전용 roscore 를 띄운다; 0 이면 기본 master 11311 을 그대로 사용)
#
# 출력: $OUT_ROOT/<cond>/attempt_N/{map_localizer.csv, map_localizer_summary.txt, *_latency.csv, launch.log, bag.log}
# 평가: python3 $ISAAC_DATA/experiments/eval_isaac_ate.py --results $OUT_ROOT --out $OUT_ROOT/eval
set -m   # roslaunch 를 별도 프로세스 그룹으로 → kill -INT 가 먹는다
WS="${WS:-/home/rain/online_static_feature_ws}"
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

ISAAC_DATA="${ISAAC_DATA:-/home/rain/data/IsaacSim_Warehouse/isaac_data}"
BAG_DIR="${BAG_DIR:-$ISAAC_DATA/online_scan_scenario_ros1}"
OUT_ROOT="${OUT_ROOT:-$ISAAC_DATA/experiments/results_ros1}"
MAP_NAME="${MAP_NAME:-dynamic_environment_1}"
MATCHER="${MATCHER:-vgicp}"
RATE="${RATE:-1.0}"
DRAIN="${DRAIN:-10}"
ROS_PORT="${ROS_PORT:-11411}"
CONFIG="${CONFIG:-localization_isaac.yaml}"
FEATURE_CONFIG="${FEATURE_CONFIG:-gis_isaac_feature.yaml}"   # voxel 0.25 m (ROS2 Isaac 실험과 동일)

ATTEMPTS=(${1:-1 2 3 4 5 6 7 8 9 10})
CONDS=(${2:-raw_dyn mav_sta})

DYN_MAP="$ISAAC_DATA/gt_map/dynamic/$MAP_NAME/gt_map.pcd"
STA_SURF="$ISAAC_DATA/gt_map/static/$MAP_NAME/gt_map_surface.pcd"
STA_CORN="$ISAAC_DATA/gt_map/static/$MAP_NAME/gt_map_corner.pcd"
for f in "$DYN_MAP" "$STA_SURF" "$STA_CORN"; do [ -s "$f" ] || { echo "missing map: $f"; exit 1; }; done

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"
RC=""
if [ "$ROS_PORT" != "0" ]; then
    export ROS_MASTER_URI="http://localhost:$ROS_PORT"
    roscore -p "$ROS_PORT" > "$OUT_ROOT/roscore.log" 2>&1 &
    RC=$!
    for i in $(seq 1 30); do rostopic list >/dev/null 2>&1 && break; sleep 1; done
fi
cleanup() { [ -n "$RC" ] && kill -INT "$RC" 2>/dev/null && wait "$RC" 2>/dev/null; }
trap cleanup EXIT

echo "maps: $MAP_NAME | matcher=$MATCHER rate=$RATE | master=${ROS_MASTER_URI:-default}" | tee -a "$LOG"

for A in "${ATTEMPTS[@]}"; do
    BAG="$BAG_DIR/attempt_$A.bag"
    INIT="$ISAAC_DATA/online_scan_scenario/attempt_${A}_gt/init_pose.txt"
    if [ ! -s "$BAG" ]; then echo "[skip] no bag: $BAG" | tee -a "$LOG"; continue; fi
    if [ ! -s "$INIT" ]; then echo "[skip] no init pose: $INIT" | tee -a "$LOG"; continue; fi
    ix=$(awk '$1=="init_x:"{print $2}' "$INIT");     iy=$(awk '$1=="init_y:"{print $2}' "$INIT")
    iz=$(awk '$1=="init_z:"{print $2}' "$INIT");     iroll=$(awk '$1=="init_roll:"{print $2}' "$INIT")
    ipitch=$(awk '$1=="init_pitch:"{print $2}' "$INIT"); iyaw=$(awk '$1=="init_yaw:"{print $2}' "$INIT")

    for COND in "${CONDS[@]}"; do
        OUT="$OUT_ROOT/$COND/attempt_$A"
        mkdir -p "$OUT"; rm -f "$OUT/map_localizer.csv" "$OUT/map_localizer_summary.txt"
        case $COND in
            raw_*) mode=accumulated; fw=1 ;;
            mav_*) mode=static;      fw=3 ;;
            *) echo "unknown condition $COND"; exit 1 ;;
        esac
        case $COND in
            *_dyn) map_args=(map_pcd_path:="$DYN_MAP" map_pcd_path2:="") ;;
            *_sta) map_args=(map_pcd_path:="$STA_SURF" map_pcd_path2:="$STA_CORN") ;;
            *) echo "unknown condition $COND"; exit 1 ;;
        esac

        echo "[$COND attempt_$A] start $(date +%H:%M:%S)" | tee -a "$LOG"
        roslaunch online_static_feature map_localizer.launch \
            config:="$CONFIG" feature_config:="$FEATURE_CONFIG" \
            source_mode:="$mode" matcher_type:="$MATCHER" \
            output_dir:="$OUT" feature_window_frames:="$fw" \
            "${map_args[@]}" \
            init_x:="$ix" init_y:="$iy" init_z:="$iz" \
            init_roll:="${iroll:-0}" init_pitch:="${ipitch:-0}" init_yaw:="$iyaw" \
            > "$OUT/launch.log" 2>&1 &
        LP=$!
        # localizer 준비 판정: 맵 로드·전처리가 끝난 뒤에야 source 토픽을 구독하므로,
        # 그 토픽의 구독자 목록에 /map_localizer 가 나타나면 준비된 것이다.
        # (launch.log 의 "map_points=" 로그는 파이프 버퍼링 때문에 수 분 늦게 찍혀서 기준으로 못 쓴다)
        case $mode in
            static)      src_topic=/online_static_feature_node/scan_static_features ;;
            accumulated) src_topic=/scan_matching_accumulator_node/dense_window_cloud ;;
        esac
        for i in $(seq 1 300); do
            rostopic info "$src_topic" 2>/dev/null | grep -q "/map_localizer" && break
            kill -0 $LP 2>/dev/null || break
            sleep 1
        done
        echo "[$COND attempt_$A] localizer ready after ${i}s" | tee -a "$LOG"
        sleep 2
        rosbag play -r "$RATE" "$BAG" --topics /point_cloud > "$OUT/bag.log" 2>&1
        sleep "$DRAIN"
        kill -INT $LP 2>/dev/null; wait $LP 2>/dev/null
        rows=$(( $(wc -l < "$OUT/map_localizer.csv" 2>/dev/null || echo 1) - 1 ))
        echo "[$COND attempt_$A] done $(date +%H:%M:%S)  localized frames: $rows  $(tr '\n' ' ' < "$OUT/map_localizer_summary.txt" 2>/dev/null)" | tee -a "$LOG"
    done
done
echo "ALL DONE $(date +%H:%M:%S)" | tee -a "$LOG"
