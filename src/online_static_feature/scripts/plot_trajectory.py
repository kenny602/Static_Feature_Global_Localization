#!/usr/bin/env python3
"""추정 궤적 PCD를 PNG로 렌더링한다 (여러 개면 겹쳐 그린다).

map_localizer_node가 저장하는 trajectory PCD와 교수님 결과(map_trajectory.pcd)는
동일 포맷(x y z intensity=frame index)이라 그대로 비교할 수 있다.

사용 예:
  ./plot_trajectory.py out/vgicp_features_trajectory.pcd \
                       ~/data/ETRI_노은역/PCD_RESULT/map_trajectory.pcd \
                       -o out/compare.png
"""
import argparse
import os
import struct
import sys


def load_pcd_xyzi(path):
    """binary/ascii PCD에서 (x, y, z) 리스트를 읽는다."""
    with open(path, 'rb') as f:
        data = f.read()

    header_end = data.find(b'DATA ')
    if header_end < 0:
        raise ValueError('%s: DATA 필드가 없음' % path)
    line_end = data.find(b'\n', header_end)
    header = data[:line_end].decode('ascii', errors='replace')
    fmt = data[header_end + 5:line_end].decode('ascii').strip()

    fields, size, count, points = [], [], [], 0
    for line in header.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == 'FIELDS':
            fields = tok[1:]
        elif tok[0] == 'SIZE':
            size = [int(v) for v in tok[1:]]
        elif tok[0] == 'COUNT':
            count = [int(v) for v in tok[1:]]
        elif tok[0] == 'POINTS':
            points = int(tok[1])

    if not count:
        count = [1] * len(fields)
    stride = sum(s * c for s, c in zip(size, count))

    if fmt == 'ascii':
        xs, ys, zs = [], [], []
        for line in data[line_end + 1:].decode('ascii').splitlines():
            tok = line.split()
            if len(tok) >= 3:
                xs.append(float(tok[0]))
                ys.append(float(tok[1]))
                zs.append(float(tok[2]))
        return xs, ys, zs
    if fmt != 'binary':
        raise ValueError('%s: 지원하지 않는 DATA 형식 (%s)' % (path, fmt))

    body = data[line_end + 1:]
    xs, ys, zs = [], [], []
    for i in range(points):
        off = i * stride
        if off + 12 > len(body):
            break
        x, y, z = struct.unpack_from('<fff', body, off)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pcd', nargs='+', help='궤적 PCD 경로 (여러 개면 겹쳐 그림)')
    ap.add_argument('-o', '--output', default='trajectory.png')
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    for path in args.pcd:
        if not os.path.exists(path):
            print('건너뜀 (없는 파일): %s' % path, file=sys.stderr)
            continue
        xs, ys, _ = load_pcd_xyzi(path)
        if not xs:
            print('건너뜀 (빈 궤적): %s' % path, file=sys.stderr)
            continue
        label = os.path.basename(path)
        ax.plot(xs, ys, linewidth=1.2, label='%s (%d)' % (label, len(xs)))
        ax.plot(xs[0], ys[0], 'o', markersize=7)   # 시작점

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Estimated trajectory')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print('saved: %s' % args.output)


if __name__ == '__main__':
    main()
