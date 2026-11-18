# 4 GB VRAM and laptop guide

Settings for running Donkey Kong Bananza on GPUs with 4 GB of VRAM and gaming laptops (for example an RTX 3050 Ti
Laptop, GTX 1650 or GTX 1050 Ti) with 6 to 8-core CPUs. The settings go in `dkb.ini` next to `dkb.exe`;
`totk.ini.example` describes every key and its default.

## The 4 GB VRAM limit

On a 4 GB card, Windows (the desktop compositor) and background programs often take 400 to 800 MB of VRAM, which leaves
the game about 3.2 to 3.5 GB.

One run reported by the author of this guide: Bananza 1.0.0 on build `0.3.0+a5796d9`, an Intel Core i7-11800H (8 cores,
16 threads), an RTX 3050 Ti Laptop GPU (4 GB, 75 W), NVIDIA driver 572.70, 16 GB of DDR4-3200, Windows 11:

- VRAM in use at the start was about 2.2 GB, at a steady 59.7 fps.
- Over 10 to 12 minutes the resident textures grew past 3.48 GB, up to the budget the log reports
  (`2929 of 3041 MiB in use`).
- From then on, pools were released under pressure, with freezes of about 1.6 s (`replay: 1648.2 ms` in the perf
  trace), and the frame rate fell from 60 to 27 fps.

If you see the same, please open an issue with your `dkb.log` and say which build you ran (`dkb.exe --version`).

## Suggested settings

```ini
fps_limit = 60

# Release cached textures earlier (the default is 90, in percent of VRAM)
tex_vram_pressure = 75

# Record big render passes on several threads (Automatic is 6 from 12 logical processors and 4 from 8 and 3 from 4, on AMD only the RX 9070 and 9070 XT on the proprietary driver)
replay_threads = 6

# Submit the GPU work in smaller parts, so the GPU starts sooner (defaults: 64 draws, 1000 us)
split_draws = 16
split_us = 100

# Presentation
vk_present = mailbox
window_mode = borderless
assume_focus = 1
perf_overlay = fps

# Output and upscaling
output_res = 1920x1080
upscaler = fsr
sharpness = 40
```

Everything else can stay at its default. In particular:

- `realtime` and `cloth_realtime` are still experimental on Bananza. Try them only on their own, and turn them off if
  something behaves oddly.
- `nx2_read_nowait` is opt-in on purpose: its background write-back can overwrite CPU writes made after the invalidate
  returns. Leave it off.
- `async_preempt` is already on by default, and `pipe_skip` is already `fullscreen`.

## What the settings do

### Texture release (`tex_vram_pressure = 75`)

When VRAM use passes this share of the card's memory, the texture cache releases cached textures that were not used
recently. The default is 90. A lower value keeps more VRAM free, so it may stop the freezes above, at the cost of
uploading some textures again later. This has not been measured on 4 GB cards yet.

### Smaller GPU submissions (`split_draws`, `split_us`)

The replay hands its Vulkan command buffer to the GPU in parts: a new part starts after this many draws between
passes, once the current part is at least this long. Smaller parts let the GPU start earlier, and they cost a little
more CPU. This does not spread draws across threads; recording big passes on several threads is `replay_threads`.

### Mailbox presentation (`vk_present = mailbox`)

Mailbox presents without tearing and without waiting for the next refresh. On a 144 Hz laptop screen, FIFO vsync at
60 fps can stutter, because 144 / 60 = 2.4 refreshes per frame. Mailbox avoids that.

### Output resolution and the 3D picture

The game draws its 3D scene at 2/3 of the output size (1280x720 at a 1920x1080 output), and the upscaler brings it
to the output. A lower output lowers the 3D scene too: `output_res = 1280x720` draws it at about 854x480, and FSR with
a high `sharpness` on so small an image rings and washes out. For a 4 GB card, `output_res = 1920x1080` with
`upscaler = fsr` is the better trade. `scene_res = full` draws the 3D scene at the full output size, which costs more
VRAM: leave it at `auto` on 4 GB cards.

### External frame generation

With `assume_focus = 1` and `window_mode = borderless`, external frame-generation tools (such as Lossless Scaling) can
capture the window without the game pausing when it loses focus.

## Laptop notes

- **Plug the laptop in.** On battery, NVIDIA Battery Boost and the power limits (often about half the GPU's power and
  15 to 25 W for the CPU) make a steady 60 fps out of reach, whatever the settings.
- **Cooling.** Use a performance or turbo cooling profile, and keep the vents clear, so the GPU does not throttle.
