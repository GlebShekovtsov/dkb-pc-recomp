// NVN 2 program stage -> the NVN 1 shape the rest of the host takes (gfx_shader.c, vk_exec_draw.c): Eglantine's
// lowering (runtime/third_party/eglantine, GPL-3.0-or-later) rewrites the Ampere code as Maxwell, laid out as TotK's
// translator reads it, and a synthetic control block carries the fields the host reads.
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shader_recompiler/frontend/ampere/program_capture.h"

extern "C" void rt_log(const char *fmt, ...);
extern "C" uint64_t gfx_shader_nvn2_hash(const uint8_t *control, const uint8_t *code);

namespace {
constexpr size_t kControl = 0x880;   // gfx_shader.c CONTROL_SIZE
constexpr size_t kC0Layout = 0x860;  // "DKC1" + compute parameter layout (gfx_shader.c NVN2_C0_LAYOUT)
constexpr uint64_t kMaxwellNop = 0x50b0000000070f00ull;
constexpr uint64_t kMaxwellExit = 0xe30000000007000full;

struct Lowered { std::vector<uint8_t> control; std::vector<uint64_t> code; uint64_t origin; };
std::mutex lock;
// Keyed by content, not by address: a game frees program memory and loads other shaders at the same place (a scene
// change), and an address key handed those the old lowering. The key is gfx_shader_nvn2_hash (the dump name).
// ponytail: kept for the process (a few thousand small stages); free with the program if memory ever matters
std::map<uint64_t, Lowered *> cache;
// Lowered code buffer -> its key, for nvn2_origin: the draw path asks it on every draw (vk_exec_draw.c
// light_quad_fault_skip), and a scan of the cache there cost most of the render thread in heavy scenes.
std::unordered_map<const uint8_t *, uint64_t> origins;

void put32(std::vector<uint8_t> &c, size_t at, uint32_t v) { std::memcpy(c.data() + at, &v, 4); }

Lowered *lower(const uint8_t *control, const uint8_t *code) {
    uint32_t code_len, w0;
    std::memcpy(&code_len, control + 0x38, 4), std::memcpy(&w0, code, 4);
    const bool gfx = ((w0 & 31) == 1 || (w0 & 31) == 2) && ((w0 >> 5) & 31) == 4;   // SPH v4
    const uint32_t hdr = gfx ? 0x80 : 0;
    if (code_len <= hdr) {
        rt_log("nvn2: stage with %#x bytes of code (header %#x, word 0 %#x): nothing to lower", code_len, hdr, w0);
        return nullptr;
    }

    std::vector<Shader::Ampere::Instruction> native;
    Shader::Ampere::ProgramCaptureState capture;
    for (uint32_t at = hdr; at + 16 <= code_len; at += 16) {
        Shader::Ampere::Instruction insn;
        std::memcpy(insn.words.data(), code + at, 16);
        native.push_back(insn);
        const auto r = capture.Observe(insn, static_cast<uint32_t>(native.size() - 1));
        if (r != Shader::Ampere::ProgramCaptureState::Result::Continue) break;
    }
    const auto lowered = Shader::Ampere::LowerToMaxwell(native, "full");
    if (!lowered) {
        rt_log("nvn2: Ampere lowering failed at +%#x: %.*s", lowered.error_offset, (int)lowered.error.size(),
               lowered.error.data());
        return nullptr;
    }

    auto *out = new Lowered;
    auto &m = out->code;
    // The lowering places instruction 0 at 0x10 modulo 32 with a scheduling word at every 32-byte boundary. Ryujinx
    // decodes from the code start (graphics: SPH+0x50, compute: 0) and takes every 32-byte boundary counted from there
    // as a scheduling word, so the code starts with a scheduling word and a NOP and instruction 0 lands 0x10 in. For
    // graphics, the 0x30 bytes the host skips and the first 0x50 bytes of the SPH come first. (Instruction 0 straight
    // at SPH+0x50 was decoded as a scheduling word and shifted every group: one real instruction in four was lost.)
    if (gfx) {
        m.assign(6, 0);
        m.resize(16);
        std::memcpy(m.data() + 6, code, 0x50);
        // DKB_NATIVE_GS (vk_exec_draw.c, native_gs_control): the portable geometry contract reads a kind-1 version-3
        // header. The fields it reads sit at the same offsets in the native header, so a lowered geometry stage is
        // relabelled.
        if (((w0 >> 10) & 15) == 4) {
            const uint32_t relabelled = (w0 & ~uint32_t{0x3ff}) | 1u | 3u << 5;
            std::memcpy(m.data() + 6, &relabelled, 4);
        }
    }
    m.push_back(0), m.push_back(kMaxwellNop);
    size_t pos = 0x10;   // from the code start
    // A full scheduling group of EXITs closes the program: Ryujinx's decoder runs on until a block ends, and a lowered
    // program whose last native instruction is a conditional exit or a self-branch would read past the buffer.
    // The compiler's end marker, an unconditional branch to itself after the last EXIT, becomes an EXIT too: Ryujinx's
    // structuring can join it to the exit path, and the empty endless loop it then emits hangs the GPU (device lost).
    for (size_t i = 0; i < lowered.instructions.size() + 4; i++) {
        if (pos % 32 == 0) m.push_back(0), pos += 8;
        uint64_t insn = i < lowered.instructions.size() ? lowered.instructions[i] : kMaxwellExit;
        if ((insn >> 52) == 0xe24 && ((insn >> 20) & 0xffffff) == 0xfffff8 && ((insn >> 16) & 15) == 7) insn = kMaxwellExit;
        m.push_back(insn), pos += 8;
    }

    // Inline constants (c[0x1]) follow the code in the NVN 2 buffer, from control +0x40 to +0x44; they go after the
    // Maxwell code, where control +0x700 points as in NVN 1. A BRX jump table in them holds native instruction offsets
    // (index * 16, as the lowering requires); the lowered BRX jumps to Maxwell addresses, so those are rewritten.
    uint32_t c1_at, c1_end;
    std::memcpy(&c1_at, control + 0x40, 4), std::memcpy(&c1_end, control + 0x44, 4);
    const uint32_t c1_off = static_cast<uint32_t>(m.size() * 8);
    if (c1_end > c1_at && c1_end - c1_at <= 0x10000) {
        std::vector<uint32_t> c1((c1_end - c1_at) / 4);
        std::memcpy(c1.data(), code + c1_at, c1.size() * 4);
        bool brx = false;
        for (const auto &insn : native) brx |= insn.Bits(0, 12) == 0x949 || insn.Bits(0, 12) == 0x958;
        // ponytail: any 16-aligned word inside the program counts as a table entry; an integer constant that happens to
        // be one would be rewritten too. Track the LDC base of each BRX if that ever bites.
        if (brx)
            for (uint32_t &w : c1)
                if (w % 16 == 0 && w / 16 < lowered.source_instruction_offsets.size()) {
                    const uint32_t i = lowered.source_instruction_offsets[w / 16];
                    // The lowering's address_of (0x50-based). Instruction i sits at address_of(i) - 0x40 from the code
                    // start, and the lowered BRX's base works out to -0x40, so the value lands on instruction i
                    // (Ryujinx FindBrxTargets: target = BRX base + table value).
                    w = 0x50 + i * 8 + ((i + 1) / 3) * 8;
                }
        c1.resize((c1.size() + 1) & ~size_t{1});
        m.resize(m.size() + c1.size() / 2);
        std::memcpy(m.data() + c1_off / 8, c1.data(), c1.size() * 4);
    }

    out->control.assign(kControl, 0);
    const uint32_t len = static_cast<uint32_t>(m.size() * 8);
    const uint32_t type = (w0 >> 10) & 15;   // SPH shader type -> NVN stage
    const uint32_t stage = !gfx ? 5 : type == 5 ? 1 : type == 4 ? 2 : type == 2 ? 3 : type == 3 ? 4 : 0;
    put32(out->control, 0x700, c1_off);
    put32(out->control, 0x704, len);
    put32(out->control, 0x714, stage);
    if (!gfx) {
        std::memcpy(out->control.data() + 0x748, control + 0x58, 12);   // local size (NVN 2 control +0x58)
        // The compute driver bank is packed per program from these counts (gfx_shader.c, "NVN 2 compute c0"):
        // images +0xb6, uniform buffers +0xb1, storage buffers +0xb2, textures +0xb3.
        // DKC1 adds the eight root-table word counts. The marker and counts enter the code/cache hash, so an older
        // DKC0 translation cannot be reused with the new parameter-block layout.
        const uint8_t lay[8] = {'D', 'K', 'C', '1', control[0xb6], control[0xb1], control[0xb2], control[0xb3]};
        std::memcpy(out->control.data() + kC0Layout, lay, sizeof lay);
        // Shared memory bytes (NVN 2 control +0x64; Bananza's upscaling network uses 0x2a88): the translator sizes the
        // workgroup's shared array from it, and with 0 every LDS read back nothing.
        std::memcpy(out->control.data() + kC0Layout + 8, control + 0x64, 4);
        std::memcpy(out->control.data() + kC0Layout + 12, control + 0xa9, 8);
    }
    return out;
}
}  // namespace

// 1 and the replacement control and code for an NVN 2 stage, 0 when it does not lower.
extern "C" int nvn2_lower(const uint8_t *control, const uint8_t *code, const uint8_t **out_control,
                          const uint8_t **out_code) {
    const uint64_t key = gfx_shader_nvn2_hash(control, code);
    if (!key) {
        uint32_t w[6];
        std::memcpy(w, control + 0x30, sizeof w);
        rt_log("nvn2: stage with a control block of unexpected sizes (+0x30 %#x +0x38 %#x +0x44 %#x, code %p): not "
               "lowered", w[0], w[2], w[5], (const void *)code);
        return 0;
    }
    std::lock_guard<std::mutex> g(lock);
    auto [it, fresh] = cache.try_emplace(key, nullptr);
    if (fresh && (it->second = lower(control, code))) {
        it->second->origin = key;
        origins.emplace(reinterpret_cast<const uint8_t *>(it->second->code.data()), key);
    }
    if (!it->second) return 0;
    *out_control = it->second->control.data();
    *out_code = reinterpret_cast<const uint8_t *>(it->second->code.data());
    return 1;
}

// The dump name (gfx_shader_nvn2_hash) of the NVN 2 stage a lowered code buffer came from, or 0. Lowered buffers live
// for the process, so their addresses never name another stage.
// Each thread keeps its answers, the empty ones too: draws ask for every stage (note_origin in gfx_shader.c for stages
// with no origin), and with parallel replay slices (vk_exec.c) the lock was taken by up to eight threads per draw. An
// answer cannot change: a lowered buffer is registered before its address is handed out, and a code address that is
// not one never becomes one, since lowered buffers are never freed.
extern "C" uint64_t nvn2_origin(const uint8_t *code) {
    static thread_local struct { const uint8_t *code; uint64_t origin; } memo[256];
    auto &m = memo[(reinterpret_cast<uintptr_t>(code) >> 4 ^ reinterpret_cast<uintptr_t>(code) >> 12) & 255];
    if (code && m.code == code) return m.origin;
    uint64_t o;
    {
        std::lock_guard<std::mutex> g(lock);
        auto it = origins.find(code);
        o = it == origins.end() ? 0 : it->second;
    }
    m.code = code, m.origin = o;
    return o;
}
