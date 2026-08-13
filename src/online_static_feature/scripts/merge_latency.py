#!/usr/bin/env python3
"""세 노드의 latency CSV를 프레임 단위로 합쳐 총 계산시간을 낸다.

한 번 실행하면 output_dir에 CSV 3개가 생긴다:
  accumulator_latency.csv  - scan matching (항상)
  feature_latency.csv      - static feature 추출 (source_mode=accumulated면 없음)
  map_localizer.csv        - map matching (항상)

셋 다 /ouster/points의 header.stamp가 그대로 전파되므로 stamp로 조인된다.
map_localizer.csv를 기준으로 left join 하고, 프레임별 단계 시간과 총합을
total.csv로 쓴다.

사용:
  python3 merge_latency.py <output_dir> [<output_dir> ...]
  python3 merge_latency.py --compare <dir1> <dir2> ...   # 조건 비교 표만 출력
"""

import argparse
import bisect
import csv
import os
import statistics
import sys

# stamp 매칭 허용 오차 [s].
# 세 노드가 std::fixed<<setprecision(6)을 쓰면 stamp가 정확히 일치한다.
# 다만 map_localizer에 그 설정이 없던 시절의 CSV는 유효숫자 6자리로 반올림되어
# (예: 2727.925284 -> 2727.93) 최대 5ms 오차가 난다. 프레임 간격이 100ms라
# 50ms 허용치면 이웃 프레임과 혼동 없이 옛 데이터도 매칭된다.
STAMP_TOL = 0.05

ACC_FILE = "accumulator_latency.csv"
FEAT_FILE = "feature_latency.csv"
LOC_FILE = "map_localizer.csv"
OUT_FILE = "total.csv"
OUT_DIR = "/home/rain/online_static_feature_ws/output/map_localizer/ETRI_노은역/static_point/small_gicp"

# 총합에 들어가는 항목. 각 노드가 자기 알고리즘에 실제로 쓴 시간만 고른다.
# 발행(publish) 시간은 시각화/노드간 전송 비용이라 제외한다.
ACC_STAGES = ["preprocess_ms", "submap_build_ms", "source_prepare_ms",
              "small_gicp_prepare_ms", "small_gicp_align_ms",
              "dense_window_build_ms"]
FEAT_STAGES = ["window_algorithm_ms"]
LOC_STAGES = ["preprocess_ms", "align_ms"]


def read_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def build_index(rows, stamp_key):
    """stamp로 정렬한 (stamps, rows). bisect 최근접 탐색용."""
    items = []
    for r in rows or []:
        s = to_float(r, stamp_key, float("nan"))
        if s != s:  # NaN
            continue
        items.append((s, r))
    items.sort(key=lambda t: t[0])
    return [t[0] for t in items], [t[1] for t in items]


def lookup(stamps, rows, stamp, tol):
    """stamp에 가장 가까운 행과 그 거리. tol 밖이면 (None, 거리)."""
    if not stamps:
        return None, float("inf")
    i = bisect.bisect_left(stamps, stamp)
    best, best_dt = None, float("inf")
    for j in (i - 1, i):
        if 0 <= j < len(stamps):
            dt = abs(stamps[j] - stamp)
            if dt < best_dt:
                best, best_dt = rows[j], dt
    if best_dt > tol:
        return None, best_dt
    return best, best_dt


def with_suffix(base, suffix):
    stem, ext = os.path.splitext(base)
    return f"{stem}{suffix}{ext}"


def merge_one(output_dir, tol=STAMP_TOL, suffix=""):
    loc_name = with_suffix(LOC_FILE, suffix)
    loc_rows = read_csv(os.path.join(output_dir, loc_name))
    if not loc_rows:
        print(f"[skip] {output_dir}: {loc_name} 없음 또는 비어 있음", file=sys.stderr)
        return None

    acc_rows = read_csv(os.path.join(output_dir, with_suffix(ACC_FILE, suffix)))
    feat_rows = read_csv(os.path.join(output_dir, with_suffix(FEAT_FILE, suffix)))

    acc_stamps, acc_data = build_index(acc_rows, "msg_stamp_sec")
    feat_stamps, feat_data = build_index(feat_rows, "msg_stamp_sec")

    has_feature = feat_rows is not None
    out_rows = []
    unmatched_acc = 0
    unmatched_feat = 0
    max_dt = 0.0

    for lr in loc_rows:
        stamp = to_float(lr, "stamp", float("nan"))
        if stamp != stamp:
            continue

        ar, acc_dt = lookup(acc_stamps, acc_data, stamp, tol)
        if has_feature:
            fr, feat_dt = lookup(feat_stamps, feat_data, stamp, tol)
        else:
            fr, feat_dt = None, 0.0
        if ar is None:
            unmatched_acc += 1
        else:
            max_dt = max(max_dt, acc_dt)
        if has_feature:
            if fr is None:
                unmatched_feat += 1
            else:
                max_dt = max(max_dt, feat_dt)

        acc_total = sum(to_float(ar, k) for k in ACC_STAGES) if ar else 0.0
        feat_total = sum(to_float(fr, k) for k in FEAT_STAGES) if fr else 0.0
        loc_pre = to_float(lr, "preprocess_ms")
        loc_align = to_float(lr, "align_ms")
        loc_total = loc_pre + loc_align

        out_rows.append({
            "frame": lr.get("frame", ""),
            "stamp": f"{stamp:.6f}",
            "matcher": lr.get("matcher", ""),
            "source_mode": lr.get("source_mode", ""),

            # 점 수 (속도 차이의 원인)
            "raw_points": ar.get("filtered_points", "") if ar else "",
            "dense_points": ar.get("dense_points", "") if ar else "",
            "feature_points": fr.get("feature_points", "") if fr else "",
            "src_points": lr.get("source_points", ""),
            "src_points_used": lr.get("src_points_used", ""),
            "map_points": lr.get("map_points", ""),

            # 단계별 시간
            "scan_matching_ms": f"{acc_total:.6f}",
            "feature_extract_ms": f"{feat_total:.6f}",
            # src_preprocess_ms: 매 프레임 "들어온 스캔"의 축약+KD-tree+공분산.
            # 맵(target) 전처리가 아니다. 맵 쪽은 시작 시 1회이고 map_prep_ms로
            # 따로 남으며 total_ms에 포함되지 않는다.
            "src_preprocess_ms": f"{loc_pre:.6f}",
            "map_align_ms": f"{loc_align:.6f}",
            "map_matching_ms": f"{loc_total:.6f}",
            "total_ms": f"{acc_total + feat_total + loc_total:.6f}",
            "map_prep_ms_once": lr.get("map_prep_ms", ""),  # 참고용, total 제외

            # 정합 품질
            "fitness": lr.get("fitness", ""),
            "converged": lr.get("converged", ""),
            "rejected": lr.get("rejected", ""),
        })

    out_path = os.path.join(output_dir, with_suffix(OUT_FILE, suffix))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"[ok] {out_path}  ({len(out_rows)} frames, stamp 최대 오차 {max_dt*1000:.2f} ms)")
    if not has_feature:
        print("     feature_latency.csv 없음 → feature_extract_ms = 0 "
              "(source_mode=accumulated: 추출 비용 자체가 없음)")
    if unmatched_acc:
        print(f"     경고: accumulator 미매칭 {unmatched_acc}/{len(out_rows)} 프레임",
              file=sys.stderr)
    if unmatched_feat:
        print(f"     경고: feature 미매칭 {unmatched_feat}/{len(out_rows)} 프레임",
              file=sys.stderr)
    return out_rows


def summarize(rows, skip_first=20):
    """앞쪽 프레임은 캐시/CPU 부스트 워밍업이라 제외하고 통계를 낸다."""
    use = rows[skip_first:] if len(rows) > skip_first else rows
    cols = ["scan_matching_ms", "feature_extract_ms", "src_preprocess_ms",
            "map_align_ms", "map_matching_ms", "total_ms"]
    out = {"frames": len(use)}
    for c in cols:
        vals = [float(r[c]) for r in use if r[c] != ""]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        out[c] = {
            "median": statistics.median(vals_sorted),
            "mean": statistics.fmean(vals_sorted),
            "p95": vals_sorted[min(len(vals_sorted) - 1,
                                   int(0.95 * len(vals_sorted)))],
        }
    rej = [r for r in use if r["rejected"] == "1"]
    out["reject_rate"] = len(rej) / len(use) if use else 0.0
    used = [float(r["src_points_used"]) for r in use if r["src_points_used"] != ""]
    out["src_points_used"] = statistics.median(used) if used else 0.0
    return out


def print_table(named_summaries):
    cols = ["scan_matching_ms", "feature_extract_ms", "src_preprocess_ms",
            "map_align_ms", "total_ms"]
    name_w = max(len(n) for n, _ in named_summaries)
    header = f"{'condition':<{name_w}} {'frames':>7} {'src_pts':>9} {'reject':>7}"
    for c in cols:
        header += f" {c.replace('_ms', ''):>18}"
    print()
    print(header)
    print("-" * len(header))
    for name, s in named_summaries:
        line = (f"{name:<{name_w}} {s['frames']:>7} "
                f"{s['src_points_used']:>9.0f} {s['reject_rate']*100:>6.1f}%")
        for c in cols:
            if c in s:
                line += f" {s[c]['median']:>10.2f}/{s[c]['p95']:<7.2f}"
            else:
                line += f" {'-':>18}"
        print(line)
    print("\n(각 칸은 중앙값/95퍼센타일 [ms], 앞 20프레임 제외)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # nargs="*" 여야 인자 0개를 허용해 default가 적용된다.
    # "+"는 최소 1개 필수라 default가 무시되고, nargs 자체가 없으면 정확히 1개 필수다.
    # default는 리스트여야 한다. 문자열이면 for d in args.dirs 가 글자 단위로 돈다.
    ap.add_argument("dirs", nargs="*", default=[OUT_DIR],
                    help=f"map_localizer.launch의 output_dir. 생략 시 {OUT_DIR}")
    ap.add_argument("--skip-first", type=int, default=20,
                    help="통계에서 제외할 앞쪽 프레임 수 (기본 20)")
    ap.add_argument("--stamp-tol", type=float, default=STAMP_TOL,
                    help=f"stamp 매칭 허용 오차 [s] (기본 {STAMP_TOL})")
    ap.add_argument("--no-common-range", action="store_true",
                    help="조건마다 재생 구간이 달라도 자르지 않고 전체를 쓴다")
    ap.add_argument("--suffix", default="_3frame",
                    help="재실행 파일 접미사. 예: --suffix _2frame 이면 "
                         "map_localizer_2frame.csv 등을 읽어 total_2frame.csv로 쓴다")
    args = ap.parse_args()

    merged = []
    for d in args.dirs:
        rows = merge_one(d, args.stamp_tol, args.suffix)
        if rows:
            # basename만 쓰면 조건들이 전부 'small_gicp'로 겹친다.
            parts = os.path.normpath(d).split(os.sep)
            label = os.sep.join(parts[-2:]) if len(parts) >= 2 else d
            merged.append((label, rows))

    if not merged:
        return

    # 조건마다 bag 재생 구간이 다르면 다른 데이터를 비교하게 된다.
    # 공통 stamp 구간으로 잘라야 같은 장면에 대한 비교가 된다.
    if len(merged) > 1 and not args.no_common_range:
        lo = max(min(float(r["stamp"]) for r in rows) for _, rows in merged)
        hi = min(max(float(r["stamp"]) for r in rows) for _, rows in merged)
        clipped = []
        for label, rows in merged:
            keep = [r for r in rows if lo <= float(r["stamp"]) <= hi]
            if len(keep) != len(rows):
                print(f"[range] {label}: {len(rows)} → {len(keep)} 프레임 "
                      f"(공통 구간 {lo:.2f}~{hi:.2f}s)")
            clipped.append((label, keep))
        merged = clipped

    print_table([(label, summarize(rows, args.skip_first))
                 for label, rows in merged])


if __name__ == "__main__":
    main()
