#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""map_localizer CSV 를 LAS 트랙의 GT 궤적과 비교해 ATE 를 계산한다.

GT: 각 트랙 CameraPos/*Camera1.PosC (MMS 후처리 산출물).
    las_pcl 이 point_pose 모드에서 각 포인트를 뺄 때 쓴 것과 같은 파일이라
    기준점이 일치한다.

Umeyama 정렬을 하지 않는다. 추정 pose 와 GT 가 이미 같은 지도 좌표계
(UTM52N - offset 348800/4026400/70) 에 있으므로, 정렬하면 실제 오차까지
흡수해서 결과가 낙관적으로 왜곡된다.

출력:
  <csv>_ate.txt        요약 통계
  <csv>_ate.csv        프레임별 오차
  <csv>_est.tum        추정 궤적 (evo 용)
  <csv>_gt.tum         GT 궤적    (evo 용)

사용:
  rosrun online_static_feature eval_ate.py <map_localizer.csv> --track 026
"""
import argparse, csv, glob, os, sys
import numpy as np

OFFSET = np.array([348800.0, 4026400.0, 70.0])
BASE = ("/home/rain/data/ETRI_노은역/new_data/"
        "모빌리티 보행지도 구축_대전/SEC001_유성구/SURV01_유성구")


def load_gt(track):
    pat = os.path.join(BASE, f"TRACK{track}", "CameraPos", "*Camera1.PosC")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"GT 궤적을 못 찾음: {pat}")
    t, xyz = [], []
    with open(files[0], errors="ignore") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            p = line.split()
            if len(p) < 5:
                continue
            t.append(float(p[1]))
            xyz.append([float(p[2]), float(p[3]), float(p[4])])
    return np.array(t), np.array(xyz) - OFFSET, files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--track", required=True, help="예: 026")
    ap.add_argument("--out-prefix", default=None)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    if not rows:
        sys.exit(f"CSV 가 비어 있음: {a.csv}")
    est_t = np.array([float(r["stamp"]) for r in rows])
    est_p = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    rejected = np.array([int(r["rejected"]) for r in rows], bool)

    gt_t, gt_p, gt_file = load_gt(a.track)

    # 시간축 점검: use_gps_stamp=false 로 돌리면 stamp 가 wall clock 이라
    # GT 와 결합이 불가능하다. 조용히 틀린 값을 내지 않도록 여기서 막는다.
    lo, hi = gt_t.min(), gt_t.max()
    inside = ((est_t >= lo - 5) & (est_t <= hi + 5)).mean()
    if inside < 0.5:
        sys.exit(
            f"stamp 가 GT 시간 범위 밖이다 (겹침 {inside*100:.0f}%).\n"
            f"  CSV stamp : {est_t.min():.1f} ~ {est_t.max():.1f}\n"
            f"  GT  범위  : {lo:.1f} ~ {hi:.1f}\n"
            f"  pcl_pub 의 use_gps_stamp 가 true 인지 확인할 것.")

    ok = (est_t >= lo) & (est_t <= hi)
    gi = np.c_[np.interp(est_t, gt_t, gt_p[:, 0]),
               np.interp(est_t, gt_t, gt_p[:, 1]),
               np.interp(est_t, gt_t, gt_p[:, 2])]
    d = est_p - gi
    e3 = np.linalg.norm(d, axis=1)
    e2 = np.linalg.norm(d[:, :2], axis=1)

    def stats(v, m):
        v = v[m]
        if v.size == 0:
            return dict(n=0)
        return dict(n=v.size, rmse=float(np.sqrt((v**2).mean())), mean=float(v.mean()),
                    med=float(np.median(v)), p95=float(np.percentile(v, 95)),
                    max=float(v.max()))

    prefix = a.out_prefix or os.path.splitext(a.csv)[0]
    L = []
    L.append(f"입력      : {a.csv}")
    L.append(f"GT        : {gt_file}")
    L.append(f"TRACK{a.track}  프레임 {len(rows)}개 (GT 구간 내 {ok.sum()}개), "
             f"reject {rejected.sum()} ({rejected.mean()*100:.0f}%)")
    L.append(f"시간 범위 : {est_t.min():.1f} ~ {est_t.max():.1f}  ({est_t.max()-est_t.min():.1f} s)")
    L.append("")
    L.append("Umeyama 정렬 없음 (추정/GT 가 동일 좌표계)")
    L.append("")
    for lab, mask in [("전체 프레임", ok), ("reject 제외", ok & ~rejected)]:
        s3, s2 = stats(e3, mask), stats(e2, mask)
        if s3["n"] == 0:
            L.append(f"[{lab}] 해당 프레임 없음"); continue
        L.append(f"[{lab}]  n={s3['n']}")
        L.append(f"   3D ATE  RMSE {s3['rmse']:.3f}  평균 {s3['mean']:.3f}  "
                 f"중앙 {s3['med']:.3f}  95% {s3['p95']:.3f}  최대 {s3['max']:.3f}  [m]")
        L.append(f"   2D ATE  RMSE {s2['rmse']:.3f}  평균 {s2['mean']:.3f}  "
                 f"중앙 {s2['med']:.3f}  95% {s2['p95']:.3f}  최대 {s2['max']:.3f}  [m]")
        L.append("")
    txt = "\n".join(L)
    print(txt)
    open(prefix + "_ate.txt", "w").write(txt + "\n")

    with open(prefix + "_ate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "stamp", "est_x", "est_y", "est_z",
                    "gt_x", "gt_y", "gt_z", "err_x", "err_y", "err_z",
                    "err_3d", "err_2d", "rejected", "in_gt_range"])
        for i, r in enumerate(rows):
            w.writerow([r["frame"], f"{est_t[i]:.6f}",
                        *[f"{v:.4f}" for v in est_p[i]], *[f"{v:.4f}" for v in gi[i]],
                        *[f"{v:.4f}" for v in d[i]], f"{e3[i]:.4f}", f"{e2[i]:.4f}",
                        int(rejected[i]), int(ok[i])])

    # evo 용 TUM (timestamp x y z qx qy qz qw). 자세는 비교 대상이 아니라 단위 쿼터니언.
    for name, P in [("_est.tum", est_p), ("_gt.tum", gi)]:
        with open(prefix + name, "w") as f:
            for i in range(len(rows)):
                if not ok[i]:
                    continue
                f.write(f"{est_t[i]:.6f} {P[i,0]:.6f} {P[i,1]:.6f} {P[i,2]:.6f} 0 0 0 1\n")
    print(f"저장: {prefix}_ate.txt / _ate.csv / _est.tum / _gt.tum")
    print(f"evo 예: evo_ape tum {prefix}_gt.tum {prefix}_est.tum -r trans_part --t_max_diff 0.05")


main()
