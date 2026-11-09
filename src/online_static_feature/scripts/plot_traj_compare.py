#!/usr/bin/env python3
"""TUM 궤적 2개를 논문용 그림으로 그린다.

    plot_traj_compare.py <raw.tum> <static.tum> <출력.png> [눈금간격]

x/y 범위를 같은 크기로 맞추고(1:1 축척) 지정 간격으로 눈금을 놓는다.
raw = 빨강, 제안(static) = 초록. 범례 없음.
"""
import sys, os
import numpy as np

# 인자를 안 주면 쓰는 기본값. 여기만 고치면 터미널에서 인자 없이 실행할 수 있다.
OUT_ROOT = "/home/rain/online_static_feature_ws/output/map_localizer/ETRI_LAS_ICRA"
DEFAULT_RAW    = f"{OUT_ROOT}/trajectory_plot/raw_point.tum"
DEFAULT_STATIC = f"{OUT_ROOT}/trajectory_plot/static_point.tum"
DEFAULT_OUT    = f"{OUT_ROOT}/trajectory_plot/traj_compare.png"
DEFAULT_STEP   = 50.0                      # 눈금 간격 [m]
# 배경 지도(BEV). 지면을 빼고 벽면만 남긴 XY 투영 격자를 미리 만들어 둔 npy.
# 없으면 배경 없이 궤적만 그린다.  make_map_bev.py 로 생성한다.
DEFAULT_BEV    = f"{OUT_ROOT}/trajectory_plot/map_bev.npy"
BEV_RES        = 0.4                       # 배경 격자 해상도 [m]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def load(p):
    a = np.loadtxt(p)
    return a[:, 1], a[:, 2]          # tx, ty

def main():
    a = sys.argv[1:]
    raw_p = a[0] if len(a) > 0 else DEFAULT_RAW
    sta_p = a[1] if len(a) > 1 else DEFAULT_STATIC
    out   = a[2] if len(a) > 2 else DEFAULT_OUT
    step  = float(a[3]) if len(a) > 3 else DEFAULT_STEP
    for f in (raw_p, sta_p):
        if not os.path.exists(f):
            sys.exit(f"입력 tum 이 없다: {f}")

    rx, ry = load(raw_p)
    sx, sy = load(sta_p)

    # x/y 를 같은 크기 범위로. 데이터 중심에서 step 배수로 확장한다.
    x0, x1 = min(rx.min(), sx.min()), max(rx.max(), sx.max())
    y0, y1 = min(ry.min(), sy.min()), max(ry.max(), sy.max())
    span = max(x1 - x0, y1 - y0)
    span = np.ceil(span / step) * step            # step 배수로 올림
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # 눈금이 step 배수 위치에 오도록 중심도 정렬
    cx = np.round(cx / step) * step
    cy = np.round(cy / step) * step
    half = span / 2 + step * 0.15                 # 궤적이 테두리에 닿지 않게 여유

    # 맵 좌표(UTM 기반 절대값) 대신 궤적 중심을 원점으로 옮긴다.
    # 눈금이 -50 / 0 / 50 ... 으로 읽혀 그림에서 거리 파악이 쉽다.
    rx, ry = rx - cx, ry - cy
    sx, sy = sx - cx, sy - cy

    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    # ── 배경: 지도 BEV ─────────────────────────────────────────────
    # 지면을 z 로 걸러낸 벽면 점들을 XY 로 투영해 2D 히스토그램으로 깐다.
    # 점을 그대로 scatter 하면 7천만 개라 렌더링이 불가능하다.
    bev = os.environ.get("BEV", DEFAULT_BEV)
    if os.path.exists(bev):
        P = np.load(bev)
        P = P[(np.abs(P[:, 0] - cx) < half) & (np.abs(P[:, 1] - cy) < half)]
        nb = int(2 * half / BEV_RES)
        H, xe, ye = np.histogram2d(P[:, 0] - cx, P[:, 1] - cy, bins=nb,
                                   range=[[-half, half], [-half, half]])
        # 점 밀도가 수십~수천으로 편차가 커서 로그 후 상위분위로 정규화한다.
        H = np.log1p(H.T)
        vmax = np.percentile(H[H > 0], 99) if (H > 0).any() else 1.0
        ax.imshow(H, origin="lower", extent=[-half, half, -half, half],
                  cmap="Greys", vmin=0, vmax=vmax, alpha=0.30,
                  interpolation="nearest", zorder=1)
        print(f"  배경 지도 {len(P):,}점 -> {nb}x{nb} 격자 @{BEV_RES} m")
    ax.plot(rx, ry, color="#d62728", lw=2.2, zorder=2)   # raw   = 빨강
    ax.plot(sx, sy, color="#2ca02c", lw=2.2, zorder=3)   # 제안  = 초록

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(True, ls=":", lw=0.6, color="0.75")
    ax.tick_params(direction="in", top=True, right=True)
    for s in ax.spines.values():
        s.set_linewidth(0.9)

    fig.tight_layout()
    for ext in ("png", "pdf"):                    # pdf 는 논문 삽입용 벡터
        f = os.path.splitext(out)[0] + "." + ext
        fig.savefig(f, dpi=300, bbox_inches="tight")
        print("저장:", f)
    print(f"  중심 (맵좌표) = ({cx:.0f}, {cy:.0f}) -> 그림에서는 (0, 0)")
    print(f"  범위 ±{half:.0f} m (동일 span {2*half:.0f} m, 눈금 {step:.0f} m)")

main()
