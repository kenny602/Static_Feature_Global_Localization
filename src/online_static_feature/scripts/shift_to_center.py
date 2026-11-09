#!/usr/bin/env python3
"""TUM 궤적과 ROS map 을 함께 평행이동해 그림 중심을 (0,0) 으로 만든다.

evo 는 축 범위/눈금을 제어하는 옵션이 없다. 대신 입력 자체를 옮겨두면
evo 가 중심 좌표계로 그리므로 눈금이 -50 / 0 / 50 으로 읽힌다.

    shift_to_center.py <출력디렉토리> <map.yaml> <t1.tum> [t2.tum ...] [--step 50]
"""
import sys, os, shutil
import numpy as np

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    step = 50.0
    if "--step" in sys.argv:
        step = float(sys.argv[sys.argv.index("--step") + 1])
    outdir, mapyaml, tums = args[0], args[1], args[2:]
    os.makedirs(outdir, exist_ok=True)

    # 모든 궤적의 bounding box 중심을 step 배수로 반올림해 원점으로 삼는다
    lo = np.full(2, np.inf); hi = np.full(2, -np.inf)
    data = {}
    for t in tums:
        a = np.loadtxt(t); data[t] = a
        lo = np.minimum(lo, a[:, 1:3].min(0)); hi = np.maximum(hi, a[:, 1:3].max(0))
    # evo 는 축 범위를 데이터에 맞춰 자동 설정하므로, 중심을 step 배수로 반올림하면
    # 그 반올림 오차만큼 그림이 한쪽으로 치우친다. 정확한 중심을 써야 0 이 화면 중앙에 온다.
    # --round-center 를 주면 step 배수로 맞춘다(축 범위를 직접 지정하는 도구용).
    c = (lo + hi) / 2
    if "--round-center" in sys.argv:
        c = np.round(c / step) * step
    print(f"중심 (맵좌표) = ({c[0]:.2f}, {c[1]:.2f}) -> (0, 0)")
    print(f"이동 후 범위 x[{lo[0]-c[0]:.0f}, {hi[0]-c[0]:.0f}] y[{lo[1]-c[1]:.0f}, {hi[1]-c[1]:.0f}]")

    for t, a in data.items():
        a[:, 1] -= c[0]; a[:, 2] -= c[1]
        d = os.path.join(outdir, os.path.basename(t))
        np.savetxt(d, a, fmt="%.6f")
        print("  ", os.path.basename(d))

    # map 은 이미지를 그대로 두고 origin 만 옮기면 된다
    src_dir = os.path.dirname(os.path.abspath(mapyaml))
    out = []
    for line in open(mapyaml):
        if line.startswith("origin:"):
            v = eval(line.split(":", 1)[1].strip())
            out.append(f"origin: [{v[0]-c[0]:.4f}, {v[1]-c[1]:.4f}, {v[2]}]\n")
        else:
            out.append(line)
            if line.startswith("image:"):
                img = line.split(":", 1)[1].strip()
                shutil.copy(os.path.join(src_dir, img), os.path.join(outdir, img))
    dy = os.path.join(outdir, os.path.basename(mapyaml))
    open(dy, "w").writelines(out)
    print("  ", os.path.basename(dy))

main()
