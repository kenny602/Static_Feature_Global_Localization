#!/usr/bin/env python3
"""map_localizer CSV -> TUM 궤적. GT 가 없어도 되며 evo 시각화용이다.

    csv_to_tum.py <csv> [출력.tum] [--skip-rejected]

TUM 형식: timestamp tx ty tz qx qy qz qw
"""
import csv, math, sys, os

# 인자를 안 주면 쓰는 기본값. 여기만 고치면 터미널에서 인자 없이 실행할 수 있다.
OUT_ROOT = "/home/rain/online_static_feature_ws/output/map_localizer/ETRI_LAS_ICRA"
DEFAULT_CSV = f"{OUT_ROOT}/static_point/best_attempt/map_localizer.csv"
DEFAULT_TUM = None          # None 이면 입력 csv 와 같은 이름의 .tum

def quat_from_rpy(r, p, y):
    cr, sr = math.cos(r * .5), math.sin(r * .5)
    cp, sp = math.cos(p * .5), math.sin(p * .5)
    cy, sy = math.cos(y * .5), math.sin(y * .5)
    return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip = "--skip-rejected" in sys.argv
    src = args[0] if args else DEFAULT_CSV
    dst = (args[1] if len(args) > 1
           else (DEFAULT_TUM or os.path.splitext(src)[0] + ".tum"))
    if not os.path.exists(src):
        sys.exit(f"입력 csv 가 없다: {src}")
    n = kept = 0
    with open(src) as f, open(dst, "w") as out:
        for row in csv.DictReader(f):
            n += 1
            if skip and row.get("rejected") == "1":
                continue
            qx, qy, qz, qw = quat_from_rpy(float(row["roll"]),
                                           float(row["pitch"]),
                                           float(row["yaw"]))
            out.write(f"{float(row['stamp']):.6f} {float(row['x']):.6f} "
                      f"{float(row['y']):.6f} {float(row['z']):.6f} "
                      f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
            kept += 1
    print(f"{os.path.basename(dst)}  {kept}/{n} pose")

main()
