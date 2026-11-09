#!/usr/bin/env bash
# eval_ate.py 가 만든 TUM 파일로 evo 를 돌린다.
#
#   run_evo.sh <prefix|csv경로> [출력디렉토리]
#   예: run_evo.sh .../TRACK026_static_vgicp/map_localizer_3frame_병렬처리.csv
#
# eval_ate.py 를 먼저 돌려 <prefix>_gt.tum / _est.tum 이 있어야 한다.
set -u

IN=${1:?prefix 또는 csv 경로가 필요하다}
PREFIX="${IN%.csv}"
GT="${PREFIX}_gt.tum"
EST="${PREFIX}_est.tum"
OUTDIR=${2:-"$(dirname "$PREFIX")/evo"}

for f in "$GT" "$EST"; do
    [ -s "$f" ] || { echo "없음: $f"; echo "먼저 eval_ate.py 를 실행할 것"; exit 1; }
done
command -v evo_ape >/dev/null || {
    echo "evo 가 설치돼 있지 않다. 설치:"
    echo "    pip3 install --user evo"
    echo "    export PATH=\$PATH:\$HOME/.local/bin"
    exit 1
}
mkdir -p "$OUTDIR"
NAME=$(basename "$(dirname "$PREFIX")")

# -a(Umeyama 정렬)를 쓰지 않는다. 추정/GT 가 이미 같은 지도 좌표계라
# 정렬하면 실제 계통 오차까지 흡수해 결과가 낙관적으로 왜곡된다.
# -r trans_part: 위치 성분만 비교 (point_pose 라 자세는 비교 대상이 아님)
COMMON=(-r trans_part --t_max_diff 0.05)

echo "=== APE (절대 위치 오차)"
evo_ape tum "$GT" "$EST" "${COMMON[@]}" \
    --save_results "$OUTDIR/${NAME}_ape.zip" \
    --plot_mode xy --save_plot "$OUTDIR/${NAME}_ape" --no_warnings || exit 1

echo
echo "=== RPE (상대 위치 오차, 1 m 구간)"
evo_rpe tum "$GT" "$EST" "${COMMON[@]}" --delta 1 --delta_unit m \
    --save_results "$OUTDIR/${NAME}_rpe.zip" \
    --plot_mode xy --save_plot "$OUTDIR/${NAME}_rpe" --no_warnings || true

echo
echo "=== 궤적 비교 그림"
evo_traj tum "$EST" --ref "$GT" "${COMMON[@]}" \
    --plot_mode xy --save_plot "$OUTDIR/${NAME}_traj" --no_warnings || true

echo
echo "저장: $OUTDIR"
echo "여러 조건을 한 표로 비교하려면:"
echo "    evo_res $OUTDIR/*_ape.zip -p --save_table $OUTDIR/compare.csv"
