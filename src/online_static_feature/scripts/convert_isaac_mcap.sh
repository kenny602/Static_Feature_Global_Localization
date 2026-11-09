#!/usr/bin/env bash
# Isaac Sim 스캔 bag(ROS2 Jazzy mcap) → ROS1 bag 변환. /point_cloud 만 담는다.
#
#   ./convert_isaac_mcap.sh [attempts]      예) ./convert_isaac_mcap.sh "1 2 3"   (기본 1..10)
#
# 환경: ISAAC_DATA (기본 /home/rain/data/IsaacSim_Warehouse/isaac_data)
#       OUT_DIR    (기본 $ISAAC_DATA/online_scan_scenario_ros1)
#
# rosbags 0.9.23(Python 3.8)은 rosbag2 metadata version 9(Jazzy)를 거부하므로,
# metadata.yaml 의 version 만 8로 바꾼 사본 디렉터리(mcap 은 심볼릭 링크)를 만들어 읽는다.
# 메시지 포맷(cdr, sensor_msgs/PointCloud2 x,y,z,intensity float32)은 그대로 읽힌다.
# /tf 는 넣지 않는다: GT 는 attempt_N_gt/gt_traj_lidar.tum 에 이미 있고,
# bag 의 odom→jackal 이 localizer 의 map→odom 과 충돌하기 때문.
set -euo pipefail
ISAAC_DATA="${ISAAC_DATA:-/home/rain/data/IsaacSim_Warehouse/isaac_data}"
OUT_DIR="${OUT_DIR:-$ISAAC_DATA/online_scan_scenario_ros1}"
ATTEMPTS=(${1:-1 2 3 4 5 6 7 8 9 10})
mkdir -p "$OUT_DIR"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/isaac_mcap_hack.XXXX")"
trap 'rm -rf "$WORK"' EXIT

for A in "${ATTEMPTS[@]}"; do
    SRC="$ISAAC_DATA/online_scan_scenario/attempt_$A"
    DST="$OUT_DIR/attempt_$A.bag"
    if [ -s "$DST" ]; then echo "[skip] $DST exists"; continue; fi
    [ -d "$SRC" ] || { echo "[missing] $SRC"; continue; }
    H="$WORK/attempt_$A"; mkdir -p "$H"
    ln -sf "$SRC"/*.mcap "$H/"
    sed 's/^  version: 9$/  version: 8/' "$SRC/metadata.yaml" > "$H/metadata.yaml"
    echo "[convert] attempt_$A → $DST"
    t0=$(date +%s)
    # 목적지 확장자가 .bag 이어야 rosbag1 로 쓴다 (.partial 같은 확장자면 rosbag2 로 판단해 실패)
    TMP="$OUT_DIR/.attempt_${A}_converting.bag"
    rm -f "$TMP"
    rosbags-convert "$H" --dst "$TMP" --include-topic /point_cloud
    mv "$TMP" "$DST"
    echo "[done] attempt_$A in $(( $(date +%s) - t0 )) s, $(du -h "$DST" | cut -f1)"
done
