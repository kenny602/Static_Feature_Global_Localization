#!/usr/bin/env bash
# LAS 트랙 하나로 localization 을 돌리고, 끝나면 ATE 까지 계산한다.
#
#   run_las_eval.sh <track> [source_mode] [matcher_type] [publish_rate] [chunk_seconds]
#   예: run_las_eval.sh 026 static vgicp 0.5 0.8
#
# 동작:
#   1) map_localizer.launch 를 백그라운드로 띄운다 (맵 로드에 시간이 걸려 먼저 시작)
#   2) pcl_pub.launch 를 포그라운드로 돌린다. LAS 를 다 읽으면 스스로 끝난다
#   3) pcl_pub 종료 시점에 파이프라인 안에 아직 처리 중인 프레임이 남아 있다.
#      그것들이 CSV 에 기록될 때까지 기다린 뒤 map_localizer 를 SIGINT 로 종료
#      (destructor 에서 trajectory PCD 를 저장하므로 강제 kill 하면 안 된다)
#   4) eval_ate.py 로 ATE 계산
set -u

# 위치인자를 먼저 변수로 받고 비운다.
# source 한 스크립트가 부모의 $1 을 그대로 물려받아
# /opt/ros/noetic/_setup_util.py 에 인자로 넘어가는 문제를 막는다.
# ${N:-값} = N번째 인자가 없으면 '값'을 쓴다.
# 터미널에서 인자를 안 넘기고 아래 값만 고쳐가며 실험해도 된다.
TRACK=${1:-030}           # 024 025 026 027 030 046 047
SOURCE_MODE=${2:-accumulated}  # static | accumulated | features
MATCHER=${3:-vgicp}            # vgicp | gicp | ndt | icp | plane_icp
PUBLISH_RATE=${4:-3.0}         # 벽시계 발행 주기(Hz). 낮출수록 프레임 유실이 준다
CHUNK_SECONDS=${5:-0.7}        # 데이터상 한 프레임 길이(초)
# per_point | first | mid | last.  per_point 는 청크 길이만큼 뭉개진다.
CHUNK_REF=${6:-last}          # last = 스윕 종료 pose 기준 deskew
set --

# las_pcl_ROS1 은 online_static_feature_ws 위에 체인해서 빌드돼 있으므로
# 이것 하나만 source 하면 두 패키지가 모두 보인다.
# (ROS_PACKAGE_PATH 를 src 로 덮어쓰면 안 된다. rospack find 는 되지만
#  roslaunch 가 devel 의 실행파일을 못 찾는다.)
source /opt/ros/noetic/setup.bash
source /home/rain/las_pcl_ROS1/devel/setup.bash
for pkg in online_static_feature las_pcl; do
    rospack find "$pkg" >/dev/null 2>&1 || {
        echo "패키지를 못 찾음: $pkg"
        echo "las_pcl_ROS1 을 online_static_feature_ws 위에 체인해서 다시 빌드할 것:"
        echo "  cd ~/las_pcl_ROS1 && rm -rf build devel"
        echo "  source ~/online_static_feature_ws/devel/setup.bash && catkin_make"
        exit 1; }
done

DRAIN_SEC=${DRAIN_SEC:-2}          # pcl_pub 종료 후 남은 프레임 처리 대기(초).
                                    # 구독 큐가 2라 실제로는 10초면 충분하고, 30은 안전 마진.

# 트랙별 초기 pose (맵 좌표계). 각 트랙 *_gps_sorted.las 의 첫 포인트 GPS time 에
# 대응하는 CameraPos/*Camera1.PosC 위치에서 offset(348800/4026400/70)을 뺀 값.
# init_yaw 는 전부 0 - point_pose 가 회전을 적용하지 않아 클라우드가 map 축을 유지한다.
case "$TRACK" in
  024) IX=430.29; IY=374.03; IZ=18.81 ;;
  025) IX=429.16; IY=482.44; IZ=19.74 ;;
  026) IX=427.54; IY=312.36; IZ=18.43 ;;
  027) IX=427.34; IY=206.79; IZ=18.07 ;;
  030) IX=337.17; IY=546.10; IZ=21.58 ;;
  046) IX=292.16; IY=547.45; IZ=21.91 ;;
  047) IX=317.76; IY=365.72; IZ=20.92 ;;
  *) echo "TRACK$TRACK 의 초기 pose 를 모른다. 스크립트에 추가할 것"; exit 1 ;;
esac
IYAW=0.0

WS=/home/rain/online_static_feature_ws
OUT="$WS/output/map_localizer/ETRI_LAS_EVAL/ATE/TRACK${TRACK}/${SOURCE_MODE}_${MATCHER}_cs${CHUNK_SECONDS}_${CHUNK_REF}"
CSV="$OUT/map_localizer.csv"   # map_localizer.launch 의 log_csv_path 와 같은 이름이어야 한다

mkdir -p "$OUT"
echo "=== TRACK$TRACK  source_mode=$SOURCE_MODE  matcher=$MATCHER"
echo "=== publish_rate=$PUBLISH_RATE  chunk_seconds=$CHUNK_SECONDS  chunk_ref=$CHUNK_REF"
echo "=== init pose=($IX, $IY, $IZ) yaw=$IYAW"
echo "=== 출력: $OUT"

cleanup() {
    [ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null
    [ -n "${RVIZ_PID:-}" ] && kill "$RVIZ_PID" 2>/dev/null
    [ -n "${LOC_PID:-}" ] && kill -INT "$LOC_PID" 2>/dev/null
    [ -n "${PUB_PID:-}" ] && kill -INT "$PUB_PID" 2>/dev/null
}
trap cleanup INT TERM

LOCLOG="$OUT/roslaunch_localizer.log"
: > "$LOCLOG"
PYTHONUNBUFFERED=1 stdbuf -oL -eL \
    roslaunch online_static_feature map_localizer.launch \
    config:=localization_las.yaml \
    source_mode:="$SOURCE_MODE" matcher_type:="$MATCHER" output_dir:="$OUT" \
    init_x:="$IX" init_y:="$IY" init_z:="$IZ" init_yaw:="$IYAW" > "$LOCLOG" 2>&1 &
LOC_PID=$!

# 맵 로드(2억 2천만 점)에 시간이 걸린다. map_cloud 토픽이 뜰 때까지 기다린다.
# aligned_source / markers 는 구독자가 있을 때만 발행된다(비용 절감).
# 그래서 RViz 는 재생이 시작되기 전에 떠 있어야 첫 프레임부터 보인다.
if [ "${RVIZ:-0}" = "1" ]; then
    echo "--- RViz 시작"
    rviz ${RVIZ_CONFIG:+-d "$RVIZ_CONFIG"} &
    RVIZ_PID=$!
fi

echo "--- 맵 로딩 대기... (2억 점대라 수십 초 걸린다)"
# rostopic 폴링은 이전 실행의 잔재나 타이밍 때문에 즉시 통과해버린다.
# map_localizer 가 setup 을 끝내고 찍는 "map_points=" 줄을 직접 기다린다.
# 이게 뜨기 전에 스캔을 재생하면 그 사이 이동한 거리가 통째로 초기 pose 오차가 된다.
READY=0
rm -f "$CSV"
for i in $(seq 1 900); do
    # map_localizer 의 initLog() 가 만드는 CSV. loadMap()+initMatcher() 가
    # 끝난 뒤에야 생기므로 "맵 준비 완료"의 확실한 신호다.
    # roslaunch 출력을 파일로 리다이렉트하면 블록 버퍼링이 걸려
    # 로그 문자열은 한참 뒤에야 나타난다. 파일 존재는 그 영향을 안 받는다.
    if [ -e "$CSV" ]; then READY=1; break; fi
    if grep -q "map_points=" "$LOCLOG" 2>/dev/null; then READY=1; break; fi
    if grep -qE "map cloud is empty|failed to load|FATAL|process has died" "$LOCLOG" 2>/dev/null; then
        echo "map_localizer 가 기동에 실패했다:"; tail -20 "$LOCLOG"; exit 1
    fi
    kill -0 "$LOC_PID" 2>/dev/null || { echo "map_localizer 가 죽었다:"; tail -20 "$LOCLOG"; exit 1; }
    sleep 2
done
[ "$READY" = "1" ] || { echo "맵 로딩이 30분 안에 안 끝났다"; tail -20 "$LOCLOG"; exit 1; }
tail -f "$LOCLOG" & TAIL_PID=$!
echo "--- 맵 준비됨. 스캔 재생 시작"

roslaunch las_pcl pcl_pub.launch track:="$TRACK" \
    publish_rate:="$PUBLISH_RATE" chunk_seconds:="$CHUNK_SECONDS" \
    chunk_ref:="$CHUNK_REF" &
PUB_PID=$!
wait "$PUB_PID"
echo "--- 스캔 재생 완료. 남은 프레임 처리 대기 ${DRAIN_SEC}s"
sleep "$DRAIN_SEC"

[ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null
echo "--- map_localizer 종료 (trajectory PCD 저장)"
kill -INT "$LOC_PID" 2>/dev/null
wait "$LOC_PID" 2>/dev/null
sleep 3

if [ ! -s "$CSV" ]; then
    echo "CSV 가 없다: $CSV"; exit 1
fi
EXPECT=$(TRACK="$TRACK" CS="$CHUNK_SECONDS" python3 "$WS/src/online_static_feature/scripts/_expected_frames.py")
GOT=$(( $(wc -l < "$CSV") - 1 ))
echo "--- 프레임: $GOT / 기대 $EXPECT"
[ "$GOT" -lt $((EXPECT * 8 / 10)) ] && \
    echo "!!! 20% 이상 유실. publish_rate 를 더 낮출 것 (현재 $PUBLISH_RATE)"

echo "--- ATE 계산"
python3 "$WS/src/online_static_feature/scripts/eval_ate.py" "$CSV" --track "$TRACK"

echo
echo "--- evo 로 그림/표까지 뽑으려면:"
echo "    $WS/src/online_static_feature/scripts/run_evo.sh \"$CSV\""
