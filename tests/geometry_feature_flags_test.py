"""CPU admission regression for independent ordinary/native GS feature flags.

Extracts actual geometry_draw, both cached feature gates, stage lookup, native
marker classification and primitive/control helpers. Original owned header/control
metadata only: no shader translation, lowering, Vulkan loader or GPU execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    at = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[at] == '{') - (source[at] == '}')
        at += 1
    return source[start:at]


def fixture(draw):
    shader = (ROOT / 'runtime/gfx_shader.c').read_text()
    pipeline = (ROOT / 'runtime/vk_pipeline.c').read_text()
    warning = draw.index('enum { W_PROGRAM')
    cache = draw.index('#define STAGEM_N ')
    declarations = draw[warning:draw.index(';', warning) + 1]
    # The per-serial stage cache, after the scan it falls back to.
    declarations += '\n' + function(draw, 'static int find_stage_scan(')
    declarations += '\n' + draw[cache:draw.index('static int find_stage(', cache)] + '\n'
    providers = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"
#include "nvn_geometry_control.h"
#define NVN_STAGE_VS NVN_SHADER_STAGE_VERTEX
#define NVN_STAGE_FS NVN_SHADER_STAGE_FRAGMENT
#define NVN_STAGE_GS NVN_SHADER_STAGE_GEOMETRY
#define NVN_STAGE_CS NVN_SHADER_STAGE_COMPUTE
static uint8_t geometry_input;
static int ordinary_enabled, fix_c0, warnings;
static const uint8_t *original_native_code;
static int geometry_supported = 1;
GfxStats gfx_counters;
int vk_geometry_supported(void) { return geometry_supported; }
int cfg_bool(const char *name) {
    if(!strcmp(name,"TOTK_NATIVE_GS")){const char *e=getenv("DKB_NATIVE_GS");return e&&*e=='1';}
    if(strcmp(name,"TOTK_EXPERIMENTAL_GEOMETRY"))abort();
    return ordinary_enabled;
}
static int c0_on(void) { return fix_c0; }
uint64_t nvn2_origin(const uint8_t *code) { return code==original_native_code?17:0; }
static void warn_once(int kind,const char *format,...) { (void)kind;(void)format;warnings++; }
'''
    functions = '\n'.join([
        function(shader, 'int gfx_shader_is_nvn2('),
        function(pipeline, 'uint8_t vk_geometry_input('),
        function(draw, 'static int find_stage('),
        function(draw, 'static int native_gs_on('),
        function(draw, 'static int experimental_geometry_on('),
        function(draw, 'static int geometry_draw('),
    ])
    cases = r'''
enum { CODE_BYTES=0x100 };
static uint8_t controls[4][0x880],codes[4][CODE_BYTES];
static prog_t program;
static VkExec execution;
static uint64_t serial;
static unsigned checks,failures;
static const char *label;
#define CHECK(c) do {checks++;if(!(c)){failures++;fprintf(stderr,"%s:%d: %s\n",label,__LINE__,#c);}}while(0)
static void put(uint8_t *p,uint32_t value) {memcpy(p,&value,4);}
static void setup(void) {
    memset(controls,0,sizeof controls);memset(codes,0,sizeof codes);memset(&program,0,sizeof program);
    memset(&execution,0,sizeof execution);original_native_code=NULL;fix_c0=0;
    const unsigned stages[]={NVN_STAGE_VS,NVN_STAGE_GS,NVN_STAGE_FS,NVN_STAGE_GS};
    for(unsigned i=0;i<4;i++) {
        put(controls[i]+0x700,CODE_BYTES);put(controls[i]+0x704,CODE_BYTES);put(controls[i]+0x714,stages[i]);
        program.sd[i].control=controls[i];program.sd[i].data=(uintptr_t)codes[i];
    }
    put(codes[1]+0x30,1u|(3u<<5)|(4u<<10));
    program.nstage=3;execution.prog=&program;execution.prog_stages=NVN_SHADER_STAGE_ALL_GRAPHICS_BITS;
}
static void expect(uint32_t primitive,int wanted,const char *name) {
    label=name;program.serial=++serial;
    uint8_t before_controls[sizeof controls],before_codes[sizeof codes];
    memcpy(before_controls,controls,sizeof controls);memcpy(before_codes,codes,sizeof codes);
    prog_t before=program;geometry_input=0xff;
    int actual=geometry_draw(&execution,primitive);
    CHECK(actual==wanted);
    CHECK(geometry_input==(wanted==1?vk_geometry_input(primitive):0));
    CHECK(!memcmp(before_controls,controls,sizeof controls));CHECK(!memcmp(before_codes,codes,sizeof codes));
    CHECK(!memcmp(&before,&program,sizeof program));
}
static void passthrough(void) {
    put(codes[1]+0x30,1u|(3u<<5)|(4u<<10)|(1u<<24));
    put(codes[1]+0x38,1u<<24);put(codes[1]+0x3c,7u<<24);put(codes[1]+0x40,1);
}
int main(int argc,char **argv) {
    if(argc!=3)return 2;
    ordinary_enabled=atoi(argv[1]);int native=atoi(argv[2]);
    const unsigned ordinary[]={NVN_DRAW_PRIMITIVE_POINTS,NVN_DRAW_PRIMITIVE_LINES,NVN_DRAW_PRIMITIVE_LINE_STRIP,
        NVN_DRAW_PRIMITIVE_LINES_ADJACENCY,NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY,
        NVN_DRAW_PRIMITIVE_TRIANGLES,NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP,NVN_DRAW_PRIMITIVE_TRIANGLE_FAN,
        NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY,NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY};
    for(unsigned i=0;i<sizeof ordinary/sizeof *ordinary;i++) {
        setup();expect(ordinary[i],ordinary_enabled?1:0,"ordinary feature composition");
    }
    setup();fix_c0=1;expect(NVN_DRAW_PRIMITIVE_TRIANGLES,ordinary_enabled?-1:0,"ordinary c0 guard");
    setup();execution.prog_stages=1u<<NVN_STAGE_VS;
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,ordinary_enabled?-1:0,"ordinary partial-bind guard");
    setup();program.nstage=4;
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,ordinary_enabled?-1:0,"ordinary duplicate-GS guard");
    setup();program.nstage=7;
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,ordinary_enabled?-1:0,"ordinary stage-count guard");
    setup();program.nstage=4;put(controls[3]+0x714,99);
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,ordinary_enabled?-1:0,"ordinary unknown-stage guard");
    setup();expect(NVN_DRAW_PRIMITIVE_PATCHES,ordinary_enabled?-1:0,"ordinary topology guard");
    setup();passthrough();uint32_t skip[8];skip[0]=0x0fffffff;
    for(unsigned i=1;i<8;i++)skip[i]=UINT32_MAX;
    label="authored portable envelope";CHECK(nvn_geometry_control_pack_portable(controls[1],0x880,codes[1],CODE_BYTES,skip));
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,ordinary_enabled?1:0,"portable ordinary composition");
    put(controls[1]+NVN_GEOMETRY_CONTROL_OFFSET+8,2);
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,ordinary_enabled?-1:0,"ordinary malformed portable metadata guard");
    setup();original_native_code=codes[1];passthrough();
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,native?1:ordinary_enabled?-1:0,"existing native subset gate");
    put(codes[1]+0x40,2);
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,native?0:ordinary_enabled?-1:0,"existing native output-bound fallback");
    setup();original_native_code=codes[1];passthrough();
    expect(NVN_DRAW_PRIMITIVE_LINES,native?0:ordinary_enabled?-1:0,"existing native topology fallback");
    program.nstage=2;
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,native?0:ordinary_enabled?-1:0,"existing native missing-FS fallback");
    setup();original_native_code=codes[1];passthrough();put(controls[1]+0x704,0x80);
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,native?0:ordinary_enabled?-1:0,"existing native short-code fallback");
    setup();original_native_code=codes[1];passthrough();geometry_supported=0;
    expect(NVN_DRAW_PRIMITIVE_TRIANGLES,native?-1:ordinary_enabled?-1:0,"native GS without a device geometry stage");
    label="geometry stage skip counter";CHECK(gfx_counters.draws_skipped==(native?1u:0u));
    geometry_supported=1;
    printf("ordinary=%d native=%d: %u checks, %u failures, %d warnings (actual admission only)\n",ordinary_enabled,native,checks,failures,warnings);
    return failures?1:0;
}
'''
    return providers + declarations + functions + cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', required=True)
    parser.add_argument('--deps', type=Path, required=True)
    parser.add_argument('--opt', choices=('0', '2'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--draw-source', type=Path, help='optional exact saved PR67 baseline source')
    args = parser.parse_args()
    out = args.out.resolve();out.relative_to((ROOT / 'build').resolve());out.mkdir(parents=True, exist_ok=False)
    source = args.draw_source or ROOT / 'runtime/vk_exec_draw.c'
    path = out / 'fixture.c';path.write_text(fixture(source.read_text()))
    exe = out / 'fixture.exe'
    command = [args.cc, '-std=gnu11', '-O'+args.opt, '-Wall', '-Wextra', '-Werror', '-I', ROOT / 'runtime',
               '-I', args.deps / 'include', '-I', args.deps / 'include/vma', path, '-o', exe]
    compiled = subprocess.run([str(v) for v in command], capture_output=True, text=True, timeout=60)
    (out / 'compile.log').write_text(compiled.stdout+compiled.stderr)
    if compiled.returncode:raise RuntimeError(compiled.stderr)
    rows=[]
    for ordinary in (0,1):
        for native in (0,1):
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('DKB_NATIVE_GS',None)
            if native:env['DKB_NATIVE_GS']='1'
            result=subprocess.run([str(exe),str(ordinary),str(native)],env=env,capture_output=True,text=True,timeout=30)
            (out/f'{ordinary}-{native}.log').write_text(result.stdout+result.stderr)
            rows.append(dict(ordinary=ordinary,native=native,exit=result.returncode,result=result.stdout.strip()))
            print(result.stdout,end='')
            if result.returncode:print(result.stderr,end='')
    report={'results':rows,'optimization':args.opt,'draw_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'providers':['original owned complete control/header buffers and actual VkExec/prog_t headers',
                         'cfg ordinary gate value and c0 state; real native getenv gate in separate child processes',
                         'native origin is a bounded pointer-to-scalar provider; no native lowering or code execution',
                         'warning counter only; actual stage scan/cache and portable metadata helpers'],
            'limits':['admission only, no translator/cache/pipeline/descriptors/Vulkan/GPU',
                      'preserves existing native subset behavior; does not establish native correctness']}
    (out/'handoff.json').write_text(json.dumps(report,indent=2)+'\n')
    return int(any(r['exit'] for r in rows))


if __name__=='__main__':raise SystemExit(main())
