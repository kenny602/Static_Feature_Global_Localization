#!/usr/bin/env python3
"""LAS 트랙 길이 / chunk_seconds = 기대 프레임 수. run_las_eval.sh 가 쓴다."""
import os, struct, glob
B = ("/home/rain/data/ETRI_노은역/new_data/"
     "모빌리티 보행지도 구축_대전/SEC001_유성구/SURV01_유성구")
p = sorted(glob.glob(os.path.join(B, f"TRACK{os.environ['TRACK']}", "Laser01",
                                  "*_gps_sorted.las")))[0]
h = open(p, 'rb').read(400)
off = struct.unpack_from('<I', h, 96)[0]
rlen = struct.unpack_from('<H', h, 105)[0]
n = struct.unpack_from('<I', h, 107)[0]
with open(p, 'rb') as f:
    f.seek(off);              t0 = struct.unpack_from('<d', f.read(rlen), 20)[0]
    f.seek(off + (n-1)*rlen); t1 = struct.unpack_from('<d', f.read(rlen), 20)[0]
print(int(round((t1 - t0) / float(os.environ['CS']))))
