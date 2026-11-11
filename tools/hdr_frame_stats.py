"""Nits statistics of a native-HDR frame dump, for checking Bananza's HDR output without an HDR display.

Run the game with DKB_HDR_NATIVE=1 (hdr = 1) and DKB_DUMP=key, press F9 in a bright scene, then:

    .venv\\Scripts\\python tools\\hdr_frame_stats.py build\\dkb458752\\dump --peak 1400 --white 203

The game's final window texture is A2B10G10R10 holding Rec. 2020 PQ code values (docs/hdr-native-plan.md);
runtime/vk_dump.c stores it as an 8-bit PNG of those code values (pass_<seq>_t0_<w>x<h>_64_rgb10a2.png). This reads
the largest such target, decodes PQ to nits and reports: luminance percentiles and maximum against the peak the game
grades to (the game's own SetLayerNextHdrExtendedContentLuminanceRangeMetadata value, which DKB_HDR_TRACE=1 logs: 1400
nits in 3.2.0 gameplay), the share of the frame above paper white, pixels pinned at the peak (clipping), the
frame-average light level, pixels outside Rec. 709 (the game grades in Rec. 709, so few should be; the 8-bit dump's
rounding can add small excursions) and a histogram.
8 of the 10 bits survive the dump: near 1000 nits one code step is a few percent, so values are close, not exact.

The dump holds images of the game: it stays under build/ and is never shared. This script prints numbers only.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# SMPTE ST 2084 (PQ) constants
M1, M2 = 2610 / 16384, 2523 / 4096 * 128
C1, C2, C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
# Rec. 2020 to Rec. 709 primaries, linear light (ITU-R BT.2087 inverse)
BT2020_TO_709 = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]])
LUMA_2020 = np.array([0.2627, 0.6780, 0.0593])


def pq_to_nits(e):
    p = np.power(np.clip(e, 0, 1), 1 / M2)
    return 10000 * np.power(np.maximum(p - C1, 0) / (C2 - C3 * p), 1 / M1)


def nits_to_pq(n):
    y = np.power(n / 10000, M1)
    return np.power((C1 + C2 * y) / (1 + C3 * y), M2)


def window_texture(dump):
    """The largest, latest A2B10G10R10 colour target of the dump: the game's HDR window texture."""
    found = []
    for p in dump.glob('pass_*_t*_*x*_64_*.png'):
        if p.name.endswith(('_tm.png', '_nan.png')):
            continue
        m = re.match(r'pass_(\d+)_t\d+_(\d+)x(\d+)_64_', p.name)
        if m:
            found.append((int(m[2]) * int(m[3]), int(m[1]), p))
    return max(found)[2] if found else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump', type=Path, help='the dump folder (DKB_DUMP_DIR, default dump next to dkb.exe)')
    ap.add_argument('--peak', type=float, default=1400, help='the peak the game grades to, in nits (default 1400)')
    ap.add_argument('--white', type=float, default=203, help='paper white, hdr_game_nits (default 203)')
    a = ap.parse_args()
    path = window_texture(a.dump)
    if not path:
        sys.exit('no A2B10G10R10 colour target in the dump: was the game in its HDR mode (DKB_HDR_NATIVE=1, hdr = 1)?')
    code = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255
    rgb = pq_to_nits(code)        # linear Rec. 2020, nits per channel
    y = rgb @ LUMA_2020
    top = rgb.max(axis=2)
    n = y.size
    pct = lambda v, q: float(np.percentile(v, q))
    share = lambda mask: 100 * np.count_nonzero(mask) / n

    print(f'frame: {path.name} ({code.shape[1]}x{code.shape[0]})')
    print(f'luminance, nits: min {y.min():.2f}  median {pct(y, 50):.1f}  p90 {pct(y, 90):.1f}  p99 {pct(y, 99):.1f}  '
          f'p99.9 {pct(y, 99.9):.1f}  max {y.max():.1f}')
    print(f'brightest channel, nits: p99.9 {pct(top, 99.9):.1f}  max {top.max():.1f}  (peak {a.peak:.0f})')
    print(f'above paper white ({a.white:.0f}): {share(y > a.white):.3f}%   above 2x: {share(y > 2 * a.white):.3f}%   '
          f'above 4x: {share(y > 4 * a.white):.3f}%')
    print(f'a channel at the peak (clipped): {share(code.max(axis=2) >= nits_to_pq(a.peak) - 1 / 255):.4f}%')
    print(f'frame-average light level (mean of the brightest channel): {top.mean():.1f} nits')
    lin709 = rgb @ BT2020_TO_709.T
    low = lin709.min(axis=2)
    print(f'outside Rec. 709 by more than 5% of luminance: {share(low < -0.05 * y):.3f}%   '
          f'by more than 15%: {share(low < -0.15 * y):.3f}%')
    edges = [0, 1, 10, 50, 100, a.white, 2 * a.white, 500, 1000, 2000, 4000, 10001]
    counts, _ = np.histogram(y, bins=edges)
    print('luminance histogram:')
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        print(f'  {lo:7.0f} - {hi:7.0f} nits  {100 * c / n:8.4f}%')


if __name__ == '__main__':
    main()
