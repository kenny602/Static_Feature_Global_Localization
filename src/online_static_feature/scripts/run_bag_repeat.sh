#!/usr/bin/env bash
# ICRA 반복 실험: bag(test1_ros1_standard.bag) 으로 static 5회 + accumulated 5회.
#
#   run_bag_repeat.sh [반복횟수] [static|accumulated|both]
#
# 각 회차마다 출력 파일에 _N attempt 접미사를 붙여 보존한다.
# localization.yaml 은 phase 전환 시 자동으로 고쳤다가 종료 시 원복한다.
set -u

N=${1:-5}
PHASES=${2:-both}
WS=/home/rain/online_static_feature_ws
YAML="$WS/src/online_static_feature/config/localization.yaml"
BAG=/home/rain/data/ETRI_노은역/bag/test1_ros1_standard.bag
BACKUP="$WS/output/localization.yaml.orig"
DRAIN=${DRAIN:-40}

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

mkdir -p "$(dirname "$BACKUP")"
cp "$YAML" "$BACKUP"
restore() { cp "$BACKUP" "$YAML"; echo "[정리] localization.yaml 원복"; }
cleanup() {
    [ -n "${LOC_PID:-}" ] && kill -INT "$LOC_PID" 2>/dev/null
    [ -n "${BAG_PID:-}" ] && kill -INT "$BAG_PID" 2>/dev/null
    restore; exit 130
}
trap cleanup INT TERM

# phase 별 localization.yaml 설정
set_static() {
    python3 - "$YAML" <<'PY'
import io,sys
p=sys.argv[1]; L=io.open(p,encoding="utf-8").read().split("\n")
L[53]="  window_frames: 3"                                    # feature 노드 누적
L[77]="  # map_pcd_path: /home/rain/data/ETRI_노은역/PCD_RESULT/raw_map.pcd"
L[78]='  # map_pcd_path2: ""'
L[80]="  map_pcd_path:  /home/rain/offline_static_feature_ws/src/output_corner_define_pcd/ETRI_노은역/ETRI_50cm_surface.pcd"
L[81]="  map_pcd_path2: /home/rain/offline_static_feature_ws/src/output_corner_define_pcd/ETRI_노은역/ETRI_50cm_corner.pcd"
io.open(p,"w",encoding="utf-8").write("\n".join(L))
PY
}
set_accumulated() {
    python3 - "$YAML" <<'PY'
import io,sys
p=sys.argv[1]; L=io.open(p,encoding="utf-8").read().split("\n")
L[53]="  window_frames: 1"                                    # 요청대로 1 (미기동이라 무영향)
L[77]="  map_pcd_path: /home/rain/data/ETRI_노은역/PCD_RESULT/raw_map.pcd"
L[78]='  map_pcd_path2: ""'
L[80]="  # map_pcd_path:  /home/rain/offline_static_feature_ws/src/output_corner_define_pcd/ETRI_노은역/ETRI_50cm_surface.pcd"
L[81]="  # map_pcd_path2: /home/rain/offline_static_feature_ws/src/output_corner_define_pcd/ETRI_노은역/ETRI_50cm_corner.pcd"
io.open(p,"w",encoding="utf-8").write("\n".join(L))
PY
}

run_one() {           # $1=source_mode  $2=output_dir  $3=회차
    local MODE=$1 OUT=$2 I=$3
    mkdir -p "$OUT"
    local LOG="$OUT/roslaunch_${I}attempt.log"; : > "$LOG"
    echo "  [${MODE} ${I}/${N}] 시작 $(date +%H:%M:%S)"

    PYTHONUNBUFFERED=1 stdbuf -oL -eL \
    roslaunch online_static_feature map_localizer.launch \
        source_mode:="$MODE" output_dir:="$OUT" > "$LOG" 2>&1 &
    LOC_PID=$!

    # 맵 로드 완료까지 대기 (이 전에 bag 을 틀면 앞부분을 놓친다)
    local CSV="$OUT/map_localizer_3frame_병렬처리.csv"
    rm -f "$CSV"
    local READY=0
    for _ in $(seq 1 900); do
        # initLog() 가 만드는 CSV 존재 = loadMap+initMatcher 완료.
        # roslaunch 출력을 파일로 리다이렉트하면 버퍼링돼서 grep 이 안 먹는다.
        [ -e "$CSV" ] && { READY=1; break; }
        grep -q "map_points=" "$LOG" 2>/dev/null && { READY=1; break; }
        grep -qE "map cloud is empty|failed to load|FATAL|process has died" "$LOG" 2>/dev/null && {
            echo "  !! 기동 실패"; tail -15 "$LOG"; kill -INT "$LOC_PID" 2>/dev/null; return 1; }
        kill -0 "$LOC_PID" 2>/dev/null || { echo "  !! 노드 사망"; tail -15 "$LOG"; return 1; }
        sleep 2
    done
    [ "$READY" = 1 ] || { echo "  !! 맵 로딩 시간초과"; kill -INT "$LOC_PID" 2>/dev/null; return 1; }
    sleep 2; grep -o "map_points=[0-9]*" "$LOG" 2>/dev/null | tail -1 | sed 's/^/     맵 /'

    rosbag play "$BAG" > "$OUT/rosbag_${I}attempt.log" 2>&1 &
    BAG_PID=$!
    wait "$BAG_PID"; BAG_PID=
    echo "     bag 재생 완료, 잔여 프레임 ${DRAIN}s 대기"
    sleep "$DRAIN"

    kill -INT "$LOC_PID" 2>/dev/null      # SIGINT 여야 destructor 가 PCD 를 저장한다
    wait "$LOC_PID" 2>/dev/null; LOC_PID=
    sleep 3

    # 회차별 이름으로 보존
    local base
    for base in accumulator_latency feature_latency; do
        [ -f "$OUT/$base.csv" ] && mv "$OUT/$base.csv" "$OUT/${base}_${I}attempt.csv"
    done
    [ -f "$OUT/map_localizer_3frame_병렬처리.csv" ] && \
        mv "$OUT/map_localizer_3frame_병렬처리.csv" "$OUT/map_localizer_${I}attempt.csv"
    [ -f "$OUT/map_localizer_3frame_병렬처리.pcd" ] && \
        mv "$OUT/map_localizer_3frame_병렬처리.pcd" "$OUT/map_localizer_${I}attempt.pcd"

    local rows=0
    [ -f "$OUT/map_localizer_${I}attempt.csv" ] && rows=$(( $(wc -l < "$OUT/map_localizer_${I}attempt.csv") - 1 ))
    echo "  [${MODE} ${I}/${N}] 완료 $(date +%H:%M:%S)  프레임 ${rows}"
}

if [ "$PHASES" = "both" ] || [ "$PHASES" = "static" ]; then
    echo "=== PHASE 1: static (feature map) ==="
    set_static
    for i in $(seq 1 "$N"); do
        run_one static "$WS/output/map_localizer/ETRI_노은역/forICRA/static_point" "$i" || echo "  회차 $i 실패, 계속"
    done
fi
if [ "$PHASES" = "both" ] || [ "$PHASES" = "accumulated" ]; then
    echo "=== PHASE 2: accumulated (raw map) ==="
    set_accumulated
    for i in $(seq 1 "$N"); do
        run_one accumulated "$WS/output/map_localizer/ETRI_노은역/forICRA/raw_point" "$i" || echo "  회차 $i 실패, 계속"
    done
fi
restore
echo "=== 전체 완료 $(date +%H:%M:%S) ==="
