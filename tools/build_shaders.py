"""Compile the runtime's own GLSL (.comp, .vert, .frag) and SPIR-V assembly shaders into C headers of SPIR-V words.

Each runtime/shaders/<name>.comp (or .vert, .frag: one name per shader) becomes runtime/<name>_spv.h, which declares `static const uint32_t <name>_spv[]`
and records the SHA-256 of the source it came from. The headers are committed, so building totk.exe needs no shader
compiler; tools/build_game.py refuses to build when a header no longer matches its source. Run this after editing a
shader. Needs glslc and spirv-val from MSYS2 UCRT64 (package mingw-w64-ucrt-x86_64-shaderc, spirv-tools).

AMD's FSR 3.1 upscaler passes (runtime/third_party/fidelityfx) are built the same way, each variant with its own defines
and the SDK's include directory (FSR3_VARIANTS): runtime/fsr3_<variant>_spv.h. So are the passes of its frame
interpolation and the optical flow under it (FSR3FI_VARIANTS, runtime/fsr3_fi_*_spv.h and runtime/fsr3_of_*_spv.h).

    python tools/build_shaders.py [--check]
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(r'C:\msys64\ucrt64\bin')   # Windows; elsewhere glslc and spirv-val come from PATH
WIN = os.name == 'nt'
TARGET = 'vulkan1.1'
STAGES = {'.comp': 'compute', '.vert': 'vertex', '.frag': 'fragment', '.glsl': 'compute'}


class Job(NamedTuple):
    """One header to make: the shader file, the header's name (<name>_spv.h declares <name>_spv), the -D defines and the
    -I directories of this build of it. A shader under runtime/shaders is one job named after its file."""
    src: Path
    name: str
    defs: tuple = ()
    inc: tuple = ()


# AMD FSR 3.1's upscaler (runtime/vk_fsr3.c). The SDK ships each pass as GLSL over its own GPU headers, and a pass is
# built in several variants by defines. The options are promises about what vk_fsr3.c hands the passes: colour in the
# display range (not HDR), motion vectors at the render size and not jittered, and the game's depth as it is (OpenGL
# style, 0 at the near plane: not reversed). The accumulate pass does not sharpen (RCAS is not used). The _nowave
# pyramids are for GPUs without subgroup quad operations.
FFX_DIR = Path('third_party') / 'fidelityfx'
FSR3_OPTIONS = ('FFX_GPU=1', 'FFX_GLSL=1',
                'FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2',
                'FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE=0',
                'FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT=0',
                'FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS=1',
                'FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS=0',
                'FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH=0',
                'FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING=0')
NOWAVE = ('FFX_SPD_NO_WAVE_OPERATIONS=1',)
FSR3_VARIANTS = [   # (header name, SDK pass, extra defines)
    ('prepare_inputs', 'prepare_inputs', ()),
    ('luma_pyramid', 'luma_pyramid', ()),
    ('luma_pyramid_nowave', 'luma_pyramid', NOWAVE),
    ('shading_change_pyramid', 'shading_change_pyramid', ()),
    ('shading_change_pyramid_nowave', 'shading_change_pyramid', NOWAVE),
    ('shading_change', 'shading_change', ()),
    ('prepare_reactivity', 'prepare_reactivity', ()),
    ('luma_instability', 'luma_instability', ()),
    ('accumulate', 'accumulate', ()),
]

# AMD FSR 3.1's frame interpolation and the optical flow it runs on (runtime/vk_fsr3fi.c), from the same SDK. The options
# are the same promises about the inputs vk_fsr3fi.c is handed, which are the upscaler's: motion vectors at the render
# size and not jittered, and the game's depth as it is (not reversed). The option names are the interpolation's own, and
# differ from the upscaler's (LOW_RES_MOTION_VECTORS here, LOW_RESOLUTION_MOTION_VECTORS there); the SDK ignores a name it
# does not read, so a misspelled one would change nothing and say nothing (tests/fsr3fi_bindings_test.py).
FI_OPTIONS = ('FFX_GPU=1', 'FFX_GLSL=1',
              'FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH=0',
              'FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS=1',
              'FFX_FRAMEINTERPOLATION_OPTION_JITTERED_MOTION_VECTORS=0')
OF_OPTIONS = ('FFX_GPU=1', 'FFX_GLSL=1')
FSR3FI_VARIANTS = [   # (header name, SDK folder, pass file, defines)
    ('fsr3_fi_setup', 'frameinterpolation', 'ffx_frameinterpolation_setup_pass.glsl', FI_OPTIONS),
    ('fsr3_fi_reconstruct_previous_depth', 'frameinterpolation',
     'ffx_frameinterpolation_reconstruct_previous_depth_pass.glsl', FI_OPTIONS),
    ('fsr3_fi_reconstruct_and_dilate', 'frameinterpolation',
     'ffx_frameinterpolation_reconstruct_and_dilate_pass.glsl', FI_OPTIONS),
    ('fsr3_fi_game_motion_vector_field', 'frameinterpolation',
     'ffx_frameinterpolation_game_motion_vector_field_pass.glsl', FI_OPTIONS),
    ('fsr3_fi_optical_flow_vector_field', 'frameinterpolation',
     'ffx_frameinterpolation_optical_flow_vector_field_pass.glsl', FI_OPTIONS),
    ('fsr3_fi_disocclusion_mask', 'frameinterpolation', 'ffx_frameinterpolation_disocclusion_mask_pass.glsl', FI_OPTIONS),
    ('fsr3_fi_game_vector_field_inpainting_pyramid', 'frameinterpolation',
     'ffx_frameinterpolation_compute_game_vector_field_inpainting_pyramid_pass.glsl', FI_OPTIONS),
    ('fsr3_fi_inpainting_pyramid', 'frameinterpolation',
     'ffx_frameinterpolation_compute_inpainting_pyramid_pass.glsl', FI_OPTIONS),
    ('fsr3_fi_interpolate', 'frameinterpolation', 'ffx_frameinterpolation_pass.glsl', FI_OPTIONS),
    ('fsr3_fi_inpainting', 'frameinterpolation', 'ffx_frameinterpolation_inpainting_pass.glsl', FI_OPTIONS),
    ('fsr3_of_prepare_luma', 'opticalflow', 'ffx_opticalflow_prepare_luma_pass.glsl', OF_OPTIONS),
    ('fsr3_of_luminance_pyramid', 'opticalflow', 'ffx_opticalflow_compute_luminance_pyramid_pass.glsl', OF_OPTIONS),
    ('fsr3_of_generate_scd_histogram', 'opticalflow', 'ffx_opticalflow_generate_scd_histogram_pass.glsl', OF_OPTIONS),
    ('fsr3_of_compute_scd_divergence', 'opticalflow', 'ffx_opticalflow_compute_scd_divergence_pass.glsl', OF_OPTIONS),
    ('fsr3_of_compute_optical_flow_v5', 'opticalflow', 'ffx_opticalflow_compute_optical_flow_advanced_pass_v5.glsl',
     OF_OPTIONS),
    ('fsr3_of_filter_optical_flow_v5', 'opticalflow', 'ffx_opticalflow_filter_optical_flow_pass_v5.glsl', OF_OPTIONS),
    ('fsr3_of_scale_optical_flow_v5', 'opticalflow', 'ffx_opticalflow_scale_optical_flow_advanced_pass_v5.glsl',
     OF_OPTIONS),
]

INCLUDE = re.compile(rb'^\s*#\s*include\s*"([^"]+)"', re.M)


def source_hash(job, seen=None):
    """SHA-256 of the shader with its defines and, recursively, of every file it #includes (resolved next to the
    including file, then in the job's include directories, as glslc does)."""
    seen = set() if seen is None else seen
    data = job.src.read_bytes().replace(b'\r\n', b'\n')   # the same on a CRLF checkout
    h = hashlib.sha256(data)
    if job.defs:
        h.update(repr(job.defs).encode())
    seen.add(job.src.resolve())
    for m in INCLUDE.finditer(data):
        for d in (job.src.parent, *job.inc):
            inc = (d / m.group(1).decode()).resolve()
            if inc.exists():
                if inc not in seen:
                    h.update(source_hash(job._replace(src=inc, defs=()), seen).encode())
                break
    return h.hexdigest()


def header_of(job, words, runtime):
    lines = [f'// Generated by tools/build_shaders.py from runtime/{job.src.relative_to(runtime).as_posix()}; do not edit.',
             f'// source sha256 {source_hash(job)}',
             '#pragma once',
             '#include <stdint.h>',
             f'static const uint32_t {job.name}_spv[] = {{']
    for i in range(0, len(words), 8):
        lines.append('    ' + ', '.join(f'0x{w:08x}' for w in words[i:i + 8]) + ',')
    lines.append('};')
    return '\n'.join(lines) + '\n'


def header_path(job, runtime):
    return runtime / f'{job.name}_spv.h'


def stale(runtime):
    """Shaders under runtime whose header is missing or records another source hash."""
    out = []
    for job in shader_sources(runtime):
        h = header_path(job, runtime)
        if not h.exists() or f'// source sha256 {source_hash(job)}' not in h.read_text(encoding='utf-8'):
            out.append(job)
    return out


def shader_sources(runtime):
    jobs = [Job(p, p.stem) for p in sorted((runtime / 'shaders').iterdir()) if p.suffix in STAGES or p.suffix == '.spvasm']
    sdk = runtime / FFX_DIR
    inc = (sdk / 'include' / 'FidelityFX' / 'gpu',)
    for name, p, extra in FSR3_VARIANTS:
        src = sdk / 'shaders' / 'fsr3upscaler' / f'ffx_fsr3upscaler_{p}_pass.glsl'
        if src.exists():   # a --runtime copy without the SDK builds without FSR 3.1's headers
            jobs.append(Job(src, f'fsr3_{name}', FSR3_OPTIONS + extra, inc))
    for name, folder, p, defs in FSR3FI_VARIANTS:
        src = sdk / 'shaders' / folder / p
        if src.exists():
            jobs.append(Job(src, name, defs, inc))
    return jobs


def compile_one(job, runtime):
    src = job.src
    with tempfile.TemporaryDirectory() as tmp:
        spv = Path(tmp) / 'out.spv'
        if src.suffix == '.spvasm':
            subprocess.run([str(BIN / 'spirv-as.exe') if WIN else 'spirv-as', '--target-env', TARGET,
                            '-o', str(spv), str(src)], check=True)
        else:
            subprocess.run([str(BIN / 'glslc.exe') if WIN else 'glslc', '-O', f'--target-env={TARGET}', f'-fshader-stage={STAGES[src.suffix]}',
                            *[f'-I{d}' for d in job.inc], *[f'-D{d}' for d in job.defs],
                            '-o', str(spv), str(src)], check=True)
        subprocess.run([str(BIN / 'spirv-val.exe') if WIN else 'spirv-val', f'--target-env', TARGET, str(spv)], check=True)
        data = spv.read_bytes()
    words = [int.from_bytes(data[i:i + 4], 'little') for i in range(0, len(data), 4)]
    header_path(job, runtime).write_text(header_of(job, words, runtime), encoding='utf-8', newline='\n')
    print(f'{header_path(job, runtime).relative_to(ROOT)}: {len(words)} words')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='only report headers that do not match their source')
    ap.add_argument('--runtime', type=Path, default=ROOT / 'runtime')
    args = ap.parse_args()
    if args.check:
        bad = stale(args.runtime)
        for j in bad:
            print(f'{header_path(j, args.runtime)}: stale, run tools/build_shaders.py')
        return 1 if bad else 0
    for job in shader_sources(args.runtime):
        compile_one(job, args.runtime)
    return 0


if __name__ == '__main__':
    sys.exit(main())
