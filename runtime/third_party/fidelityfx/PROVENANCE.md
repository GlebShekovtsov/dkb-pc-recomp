# FidelityFX SDK (FSR 3.1 upscaler part)

Unmodified copy of AMD's FidelityFX SDK v1.1.4, commit `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`
(https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK), taken from Hookshot's `third_party/fidelityfx-sdk`
(https://github.com/NotRexed/Hookshot @ `fab8cb37641714bd839ae83c2a7bc582bedc4d70`). License: MIT, `LICENSE.txt`.

What is here: `include/FidelityFX/gpu/` (the core headers, `spd/`, `fsr3upscaler/`) and `shaders/fsr3upscaler/` (the
passes, without RCAS: the renderer does not sharpen with it). `tools/build_shaders.py` compiles the passes (FSR3_VARIANTS)
into `runtime/fsr3_*_spv.h`. Nothing here is edited.

The frame interpolation and optical flow files of the same SDK commit were added for `runtime/vk_fsr3fi.c`:
`include/FidelityFX/gpu/frameinterpolation/`, `include/FidelityFX/gpu/opticalflow/` (pass bodies and GLSL callbacks),
`shaders/frameinterpolation/` (ten passes) and `shaders/opticalflow/` (seven passes), from `sdk/include/FidelityFX/gpu/`
and `sdk/src/backends/vk/shaders/` of AMD's repository. Each was compared byte for byte with AMD's file at that commit.
`tools/build_shaders.py` compiles them (FSR3FI_VARIANTS) into `runtime/fsr3_fi_*_spv.h` and `runtime/fsr3_of_*_spv.h`. FSR 1
and the HLSL callbacks are still left out.
