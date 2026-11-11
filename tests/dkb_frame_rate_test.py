"""Check Bananza timing patches on synthetic state, without game files.

Exercise the actual replacement words through the C lifter at O0 and O2.
Check fractional timing, controller outputs, register preservation and memory bounds.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'recomp'))
from lifter import Lifter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', default='clang')
    ap.add_argument('--out', type=Path, default=ROOT/'build/dkb-frame-rate-test')
    ap.add_argument('--table', type=Path, default=Path('recomp/patches-dkb100.txt'), help='patch table (relative to the repo root)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lifter = Lifter(0x7000000000, simd='c')
    bodies, offsets = [], []
    for line in (ROOT/args.table).read_text().splitlines():
        fields = line.split('#', 1)[0].split()
        if fields[:3] != ['site', 'realtime', 'main']:
            continue
        pc = int(fields[3], 16)
        words = fields[5:]
        if len(words) == 1 and not words[0].startswith('@') and int(words[0], 16) >> 26 == 0x25:
            # A host call (runtime/frame_rate.c blur_frames, 3.2.0): one BL to patch page +0x250, which recomp/layout.py
            # puts at main+0x2810000. The host function itself is not exercised here.
            imm = int(words[0], 16) & 0x3FFFFFF
            assert pc + ((imm ^ 0x2000000) - 0x2000000) * 4 == 0x2810000 + 0x250, line
            continue
        offsets.append(pc)
        code = []
        for word in fields[5:]:
            if word.startswith('@page:x'):
                code.append(f'c->x[{int(word[7:])}] = (uintptr_t)page;')
            else:
                ins = lifter.lift(int(word, 16), 0x7000000000+pc)
                assert ins.kind == 'op'
                code.append(ins.code)
        bodies.append(f'void patch_{"exit" if not bodies else len(bodies)}(Cpu *c, void *page) {{\n'+ '\n'.join(code)+'\n}')
    assert len(bodies) == 8
    # The common exit must retain the controller-selected interval, not force 1: no site may replace the
    # interval load right after the first (common-exit) site.
    assert offsets[0] + 8 not in offsets
    original = lifter.lift(0xb9404281, 0x70001ead2c)
    bodies.append('void original_interval(Cpu *c) {\n'+original.code+'\n}')
    source = '#include "cpu.h"\n#include <math.h>\n#include <stdio.h>\n'+'\n'.join(bodies)
    source += r'''
int main(void) {
    unsigned errors=0, cases=0;
    for (unsigned rate=15;rate<=360;rate++) for(unsigned fraction=0;fraction<4;fraction++) {
        _Alignas(16) unsigned char page[4096]={0}, state[128], expected[128], stack[64];
        float ratio=60.0f/(rate+fraction*0.25f);
        uint32_t rounded=(uint32_t)lroundf(rate+fraction*0.25f);
        memcpy(page+8,&ratio,4); memcpy(page+12,&rounded,4);
        memset(state,0xa5,sizeof state);
        uint32_t interval=1+fraction;
        memcpy(state+0x40,&interval,4); memcpy(expected,state,sizeof state);
        Cpu c={0};
        for(unsigned i=0;i<32;i++) { c.x[i]=0x123456789abc0000ull+i; c.v[i]=(__uint128_t)c.x[i]<<64|~c.x[i]; }
        c.sp=(uintptr_t)(stack+sizeof stack); c.nzcv=0xa0000000; c.x[20]=(uintptr_t)state;
        Cpu want=c;
        want.x[0]=c.x[19]; want.x[1]=rounded; want.x[8]=(uintptr_t)page;
        memcpy(expected+0x50,&rounded,4); memcpy(expected+0x54,&rounded,4);
        patch_exit(&c,page);
        errors+=memcmp(&c,&want,sizeof c)!=0;
        errors+=memcmp(state,expected,sizeof state)!=0;
        cases++;
        want=c; want.x[1]=interval;
        original_interval(&c);
        errors+=memcmp(&c,&want,sizeof c)!=0;
        errors+=memcmp(state,expected,sizeof state)!=0;
        cases++;
        void (*clocks[])(Cpu*,void*)={patch_1,patch_2,patch_3,patch_4,patch_5,patch_6,patch_7};
        for(unsigned clock=0;clock<7;clock++) {
            memset(stack,0xa5,sizeof stack);
            c.x[8]=(uintptr_t)state; want=c;
            uint32_t bits; memcpy(&bits,&ratio,4); want.v[clock==4 ? 8 : 0]=bits;
            clocks[clock](&c,page);
            errors+=memcmp(&c,&want,sizeof c)!=0;
            errors+=memcmp(state,expected,sizeof state)!=0;
            for(unsigned i=0;i<sizeof stack-16;i++) errors+=stack[i]!=0xa5;
            cases++;
        }
    }
    printf("Bananza frame timing: %u cases, %u failures\n",cases,errors);
    return errors!=0;
}
'''
    path = args.out/'timing.c'
    path.write_text(source)
    for opt in ('-O0', '-O2'):
        exe = args.out/('timing'+opt[1:]+('.exe' if sys.platform == 'win32' else ''))
        subprocess.run([args.cc, opt, '-march=x86-64-v3', '-fno-strict-aliasing',
                        '-I', str(ROOT/'runtime'), str(path), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()
