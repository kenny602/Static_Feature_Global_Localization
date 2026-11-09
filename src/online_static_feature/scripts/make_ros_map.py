#!/usr/bin/env python3
"""포인트클라우드 맵 -> ROS occupancy grid (.pgm + .yaml).

evo_traj/evo_ape 의 --ros_map_yaml 로 배경에 깔 수 있게 만든다.
지면을 z 로 걸러 벽면만 남긴 뒤 XY 로 투영(BEV)한다.

    make_ros_map.py <map.pcd> <출력prefix> [해상도] [지면위 하한] [상한]
"""
import sys, os
import numpy as np

def main():
    src    = sys.argv[1]
    prefix = sys.argv[2]
    res    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    zlo    = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    zhi    = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0

    with open(src, "rb") as f:
        n = None
        while True:
            l = f.readline()
            if l.startswith(b"POINTS"): n = int(l.split()[1])
            if l.startswith(b"DATA"):   break
        off = f.tell()
    a = np.memmap(src, dtype=np.float32, mode="r", offset=off, shape=(n, 4))

    # 1차: 전체를 훑어 지면 높이와 XY 범위를 구한다
    zs, lo, hi = [], np.full(2, np.inf), np.full(2, -np.inf)
    for i in range(0, n, 10_000_000):
        c = np.asarray(a[i:i+10_000_000, :3])
        zs.append(c[::50, 2])
        lo = np.minimum(lo, c[:, :2].min(0)); hi = np.maximum(hi, c[:, :2].max(0))
    ground = np.percentile(np.concatenate(zs), 2)
    print(f"점 {n:,}  지면(2%)={ground:.2f}  z 창 [{ground+zlo:.2f}, {ground+zhi:.2f}]")

    W = int(np.ceil((hi[0] - lo[0]) / res))
    H = int(np.ceil((hi[1] - lo[1]) / res))
    grid = np.zeros((H, W), np.int32)
    kept = 0
    for i in range(0, n, 10_000_000):
        c = np.asarray(a[i:i+10_000_000, :3])
        m = (c[:, 2] > ground + zlo) & (c[:, 2] < ground + zhi)
        c = c[m]; kept += len(c)
        if not len(c): continue
        gx = np.clip(((c[:, 0] - lo[0]) / res).astype(np.int32), 0, W-1)
        gy = np.clip(((c[:, 1] - lo[1]) / res).astype(np.int32), 0, H-1)
        np.add.at(grid, (gy, gx), 1)
    print(f"벽면 {kept:,}점 -> {W}x{H} 격자 @{res} m")

    # 점 밀도를 0(검정=점유) ~ 254(흰색=자유) 로. 로그+분위 정규화로 윤곽을 고르게.
    L = np.log1p(grid.astype(np.float64))
    vmax = np.percentile(L[L > 0], 99) if (L > 0).any() else 1.0
    img = np.clip(1.0 - L / vmax, 0.0, 1.0)
    img = (img * 254).astype(np.uint8)
    img[grid == 0] = 254                      # 점 없는 곳은 자유공간
    img = np.flipud(img)                      # PGM 은 위에서 아래로 저장

    pgm = prefix + ".pgm"
    with open(pgm, "wb") as f:
        f.write(b"P5\n" + f"{W} {H}\n255\n".encode())
        f.write(img.tobytes())
    with open(prefix + ".yaml", "w") as f:
        f.write(f"image: {os.path.basename(pgm)}\n"
                f"resolution: {res}\n"
                f"origin: [{lo[0]:.4f}, {lo[1]:.4f}, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    print(f"저장: {pgm}  ({os.path.getsize(pgm)/1e6:.1f} MB)")
    print(f"저장: {prefix}.yaml   origin=({lo[0]:.1f}, {lo[1]:.1f})")

main()
