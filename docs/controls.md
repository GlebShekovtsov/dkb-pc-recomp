# Controls

> [!NOTE]
> This page comes from TotK Recomp. DKB Recomp uses the same input code, so it applies to Bananza too; setting and file
> names are written below with the TotK Recomp names. In the Bananza build read `TOTK_*` as `DKB_*` (both work, `DKB_`
> wins) and `totk.ini` as `dkb.ini` (a `totk.ini` is copied to it on the first run).

Input comes from the selected SDL3 gamepad, the keyboard and the mouse, merged into one Switch controller (`runtime/input.c`). Buttons are ORed; each stick takes the source with the larger deflection. Keys and mouse buttons count only while the game window is in the foreground. `TOTK_INPUT=none` disables both (automated runs). `TOTK_*` settings can also be set in `totk.ini` next to `totk.exe` (`input = none`, `motion = 0`, `rumble = 0`); the environment overrides the file. `TOTK_RUMBLE_STRENGTH` is the rumble strength in percent (0 to 300, default 130; 0 is the same as `rumble = 0`). `TOTK_RUMBLE_CURVE` is the exponent applied to the amplitude before the strength (0.2 to 1.5, default 0.6; lower makes weak pulses stronger, 1 is linear). `TOTK_RUMBLE_TRIGGERS=1` also drives the impulse triggers of an Xbox pad. `TOTK_RUMBLE_HD` (`auto` or `off`, default `off` since three third-party Switch Pro pads broke with it: stuck vibration, slow or held buttons, or a dropped link, #363, #383): with `auto`, on a Switch Pro Controller or Joy-Con the game's amplitude and frequency pairs go to the pad as an HD rumble packet (SDL's `SDL_SendGamepadEffect`, strength and curve applied to the amplitudes) instead of the two-motor `SDL_RumbleGamepad` path; `off` keeps that path. A pad that disconnects after HD packets falls back to the two-motor path for the rest of the session, because some Switch-mode clone pads drop the link at their first HD packets (#363). Other pads are unchanged.

## Controller device

Choose **Controller Device** in F1 > Controls > Controller. Changes apply to gameplay and the menu without restarting.
At startup, a saved manual choice is used when available; otherwise the first available controller is used.
A newly connected controller takes over automatically, including after a manual choice. If the active controller
disconnects, another available controller takes over. With no controllers, keyboard and mouse remain available.
Only the currently active controller controls gameplay and the controller shortcuts.

A manual menu choice applies immediately and is saved for the next startup. Automatic takeovers and fallbacks
show **Auto: controller name** and do not overwrite that saved startup preference. Choosing Auto explicitly
resets startup selection to the first available controller. Refreshing the list, opening the menu or changing
an unrelated setting does not count as a new connection. Devices already connected at launch are a baseline;
if several arrive between polls, the last enumerated new device is chosen. Discovery may take about one second.

Identities are cached per connected SDL instance ID: routine scans do not repeatedly initialize USB/Bluetooth
controllers. Failed opens are also remembered until that device disconnects; reconnecting allows a fresh attempt.
A new/reconnected device can still cause a brief initialization hitch.

The selection is saved as `pad_device` in `totk.ini` (`dkb.ini` in Bananza). Set it to `auto` to reset it, or use
`TOTK_PAD_DEVICE` / `DKB_PAD_DEVICE` to override it. The other values are opaque device keys written by the menu.
Devices with serial numbers can be identified across sessions; otherwise identical devices use fallback ordinals,
which can become ambiguous if their connection order changes. Duplicate names have numbered menu labels.

Motion and vibration use the selected controller's existing capabilities. Switching devices keeps the existing
button mappings and button-layout setting.

`TOTK_PAD_LAYOUT` (`pad_layout`, the *Button layout* row on the Controls tab, live) picks which face button is Switch A:

- `nintendo`: by position, as on a Switch (the table below).
- `xbox`: south A, east B, west X, north Y, as an Xbox pad labels them; on a PlayStation pad cross is A, circle B, square X, triangle Y.
- `auto` (default): `nintendo` for a pad whose east button SDL reports labelled A (Switch Pro Controller, Nintendo-layout pads), `xbox` for the others.
- `xbox_prompts`: the buttons as `nintendo`, and the game's prompts name the Xbox labels (see Xbox button prompts below).

The log names the layout in effect (`input: gamepad layout ...`); the menu shows it under *Automatic*. With `nintendo`, `xbox` and `auto` the game's button prompts stay Switch letters; `xbox_prompts` changes them.

| Switch | Gamepad (by position, Nintendo layout) | Keyboard and mouse | Action in the world |
|---|---|---|---|
| Left stick | left stick | W A S D | move |
| Right stick | right stick | mouse; arrow keys as a second source | camera |
| Y | west | left mouse button | attack |
| ZL | left trigger (over 50%) | right mouse button, Z | shield, lock on |
| ZR | right trigger (over 50%) | F, X | bow |
| X | north | Space | jump |
| B | south | Left Shift | run, cancel |
| A | east | E | interact, pick up |
| L | left shoulder | Q | abilities |
| R | right shoulder | R | throw weapon |
| Left stick click | left stick click | Left Ctrl | crouch |
| Right stick click | right stick click | C | first-person view |
| D-pad left, up, right, down | D-pad | 1 2 3 4 | quick-select |
| Plus | Start | Tab | menu, inventory |
| Minus | Back | M | map |

Esc is unbound in the game; it opens the settings menu and releases the mouse cursor (see Mouse camera). Z and X duplicate ZL and ZR so menu pages turn with two neighbouring keys; the menu tabs (L, R) are Q and R. The mouse wheel is unbound. The bindings are the tables of `input_map` in `runtime/input_map.c`: the keys (platform-neutral key ids) and one gamepad table per layout (gamepad inputs by position); `runtime/input_win.c` translates Win32 virtual keys and SDL buttons.

## Xbox button prompts

`pad_layout = xbox_prompts` (*Xbox prompts (A right)* on the Controls tab) is for an Xbox pad played by position. The buttons work as with `nintendo`: the right button, labelled B, does Switch A (interact, confirm), and the bottom one, labelled A, does Switch B (run, cancel). The letters the game shows are swapped to match the labels, A with B and X with Y, so each prompt names the button to press. The other layouts leave the prompts alone: with `xbox` the Switch letters already match the labels.

The swapped letters are part of the game's textures, fonts and messages, so they follow the layout and not the bindings. A button remapped with a `bind_` line or on the Controls tab keeps the letter the game draws for it, and the prompt then names the wrong button. Use `xbox_prompts` with the default bindings.

The buttons change at once, but the game reads its UI archive and fonts once at boot, so the prompts change at the next start (the menu says *prompts change after restart*). `xbox_prompts` loads `romfs-patches-xbox` next to `romfs-index.bin` in place of `romfs-patches` (`runtime/romfs.c`, `overlay_init`): the same 60 fps patches, with the swap made from the game's own files by `tools/make_romfs_patches.py --xbox-prompts`, which the setup runs next to the plain one:

- the action prompt (`DoCommand_00`) draws the letter images `Nt_KeyTexA/B/X/Y_00`, whose image data is swapped, in a diamond that names the other face buttons. The diamond's letters are text the game takes from `LayoutMsg/DoCommand_00.msbt` of each language's `Mals` archive (and the layout's own text panes behind it); both are swapped. The diamond's shape shows the button's position and stays.
- every other prompt is text in the button font (`scft/External.bfotf` in each `Font/Font*.Nin_NX_NVN.bfarc.zs`, glyphs U+E040 to U+E043); the A and B, X and Y glyphs are swapped.

Without the folder (an install made by an older setup, or one where the setup could not prepare it and showed a warning) the menu says the files are missing, the log has `RomFS overlay: pad_layout = xbox_prompts, but ...`, and the Switch prompts stay; running the setup again on the install folder writes it. In a source build, `.venv-py314\Scripts\python tools\make_romfs_patches.py --version <ver> --xbox-prompts` writes `build\<ver>\romfs-patches-xbox`.

## PlayStation button prompts (Bananza)

With a DualShock 4 or DualSense pad connected when the game starts, Bananza shows PlayStation symbols in its button prompts: a cross, circle, square and triangle for the face buttons, L1, R1, L2 and R2 for the shoulder buttons and triggers, and the Options and Create marks for + and -. `pad_prompts` in `dkb.ini` chooses: `auto` (default) as above, `playstation` always, `switch` never. The game reads its fonts once at boot, so a pad connected later, or a changed setting, takes effect at the next start. The log says `RomFS overlay: PlayStation button prompts` when they are on.

Which button each symbol names follows `pad_layout`, as the buttons do:

- `auto` (a Sony pad counts as `xbox`) and `xbox`: Switch A is the bottom button, so A shows the cross, B the circle, X the square and Y the triangle. The face-button diamonds that point at one button point at the bottom one for A, and so on.
- `nintendo` and `xbox_prompts`: Switch A is the right button, so A shows the circle, B the cross, X the triangle and Y the square.

As with the Xbox prompts, a button remapped with a `bind_` line keeps the symbol of the Switch button it was. `auto` looks at every pad connected at start, not at the one `pad_device` chooses (the choice is made later, when input starts): with a PlayStation pad connected next to the one you play with, set `pad_prompts = switch`.

Every prompt is a glyph of one font, `scft/NinDev-HardKey.bfotf`, the same in each `Font/Font*.Nin_Ounce_NVN2.bfarc.zs`: an irregular plate with the button's letter cut out of it (`tools/dkb_prompt_survey.py` found it). The setup runs `tools/dkb_ps_prompts.py`, which keeps each plate of the user's own font and cuts the PlayStation symbol into it instead of the letter: the symbols, the 1 and the 2 are drawn by the tool, and the L and R are the game's, from its ZL and ZR glyphs. It writes `romfs-patches-ps` and `romfs-patches-ps-nintendo` next to `romfs-index.bin`; `runtime/romfs.c` (`ps_prompts_init`) loads one of them next to the other overlays, which replace no Font archive. The D-pad, the sticks, Home and the controller pictures stay as the game draws them.

## Window keys

| Key | Gamepad | Action |
|---|---|---|
| F1 or Esc | Guide / Home button, or Minus + Plus together (default) | open or close the settings menu; the game gets no input while it is open |
| F10 | L3 + R3 (both stick clicks) | performance overlay: off, frame rate, full panel, full panel with the frame time broken into its parts, counter |
| F11 | Hold Start + Back for 0.5 seconds (opt-in) | windowed, or back to the last full-screen mode (`window_mode`), saved to `totk.ini` |

By default, the original immediate Minus + Plus settings shortcut remains available and follows the current bindings.
Enable **Fullscreen hold shortcut** in F1 > Controls > Controller (`pad_fullscreen_hold = 1`, or
`TOTK_PAD_FULLSCREEN_HOLD` / `DKB_PAD_FULLSCREEN_HOLD`) to replace that combination with fullscreen.
The setting defaults to off and takes effect immediately. Guide / Home and keyboard F1 work in either mode.
In Steam Deck Game Mode, enabling the fullscreen hold leaves only keyboard F1 to open settings when Steam intercepts Guide / Home.

The opt-in fullscreen hold uses the selected controller's physical SDL Start and Back buttons (Options + Create on DualSense), independently of button remapping. It fires once per hold; release both buttons before repeating. Once the buttons overlap, neither reaches the game until both are released. An individual button pressed before the combination forms still works normally. Switching or reconnecting controllers does not inherit a held shortcut.

The read-only **Shortcuts** section above **Buttons** shows settings, fullscreen and performance shortcuts using the selected controller's button labels and the enabled shortcut mode. Performance data follows the current stick-click bindings, as before. Guide / Home must be forwarded by the operating system or streaming client.

In the menu the arrow keys, the D-pad or the left stick move, Enter or A selects, Q and E or L and R switch tabs, Esc or B goes back (Switch button names, as in the table above).

## Remapping

Under *Controls*, open *Controller* to remap controller buttons or *Mouse & Keyboard* to assign up to two keys or mouse buttons per action, including movement and camera directions. Click a binding or activate it with A or Enter, then press the new input. Esc or five seconds without a press cancels, and a controller button cancels a key binding. Delete, Backspace or X clears the selected binding. A key already in use moves to the new action, and a controller button swaps with the one it replaces. Remapped face buttons (A B X Y) stay in the same physical positions when changing the button layout. *Reset all bindings* restores both devices to their defaults. F1, F8, F10, F11 and Esc are reserved. The menu displays punctuation keys using the current keyboard layout.

The tab writes `totk.ini` lines, one per Switch button or stick direction: `bind_<button> = <entry>, <entry>, ...`. The button is `A B X Y L R ZL ZR PLUS MINUS LEFT UP RIGHT DOWN LCLICK RCLICK`, or a stick direction `LSTICK_LEFT LSTICK_UP LSTICK_RIGHT LSTICK_DOWN` (movement, W A S D by default) and `RSTICK_...` (camera, the arrow keys). An entry is `pad:<name>` or `key:<name>`:

- pad names: `south east west north l r zl zr start back lstick rstick dpad_left dpad_up dpad_right dpad_down lpaddle1 rpaddle1 lpaddle2 rpaddle2 touchpad` (`zl`, `zr` are the triggers; the paddles are the back buttons of an Xbox Elite, DualSense Edge or Steam Controller, upper pair 1, lower pair 2; `touchpad` is the DualSense and DualShock 4 touchpad click; no Switch button has the paddles or the touchpad by default)
- key names: `a` to `z`, `0` to `9`, `f2` to `f7`, `f9`, `f12`, `num0` to `num9`, `space tab enter backspace lshift rshift lctrl rctrl lalt capslock insert delete home end pageup pagedown up down left right`, `num_add num_sub num_mul num_div num_decimal`, `minus equals lbracket rbracket semicolon apostrophe comma period slash backslash grave`, `mouse_l mouse_r mouse_m mouse_x1 mouse_x2`
- `key:default` and `pad:default` keep that part's defaults (`pad:default` is each layout's own button).
- `none` leaves the button unbound.

A line replaces every default of that button, pad and keyboard, so list the keys you still want or use `key:default`. Pad names are physical positions, so a line means the same in both layouts and replaces the layout's own button for that Switch button. Stick directions take keys only: the controller's sticks and the mouse camera always work. Buttons without a line keep the table above. Names are case-insensitive. An invalid entry logs one `input: bind ...` line and is skipped; a line without a valid entry changes nothing. The `TOTK_BIND_<button>` environment variables work like any other setting. A changed line applies at once (`runtime/input_win.c` builds the tables again at its next poll).

A pad that reports its face buttons swapped (A/B and X/Y, as some GuliKit models do) is fixed with:

```
bind_A = pad:south, key:E
bind_B = pad:east, key:lshift
bind_X = pad:west, key:space
bind_Y = pad:north, key:mouse_l
```

## Mouse camera

Enable **Direct mouse camera** in F1 > Controls to use raw mouse distance for rotation in Bananza 1.0.0's normal ground-follow camera: each count turns `0.064 * TOTK_MOUSE_SENS` degrees. It is off by default. The toggle and sensitivity apply live. This bypasses the stick's dead zone, acceleration and smoothing, so small movements register and fast turns are not capped at full-stick speed. The game's camera inversion settings and vertical limits still apply, and the native camera handles its orbit and collision. Vertical mouse movement can interrupt the native pitch-adjustment waiting states, just as stick input does. Controller and arrow-key input retain the game's normal behavior when the mouse is idle.

Other camera modes use the existing mouse-to-stick fallback. Each presented frame converts raw counts into `counts per second * TOTK_MOUSE_SENS / 3000` of full stick deflection, held for that frame. In that path the game's stick curve and dead zone still apply, and the larger deflection wins between mouse, controller and arrow keys. Aiming and special transformations still need gameplay validation. The setting is stored as `mouse_camera` in `totk.ini`; `DKB_MOUSE_CAMERA=0|1` overrides it. Direct rotation requires regenerating the game code with `recomp/patches-dkb100.txt`.

`TOTK_MOUSE_SENS` (0 to 10, default 1; higher turns faster, 0 turns the mouse camera and the cursor capture off). While the window is the foreground window the cursor is hidden and confined to the client area. It is released on focus loss, on minimise, while F11 changes the window, during a size or move loop and while a text entry is open. Esc releases it by hand: the cursor shows, the mouse camera and mouse buttons stop, and the window can be moved, minimised or closed. A mouse click inside the client area captures it again; that click does not reach the game. Activating the window by a click on its title bar or border does not capture, so it can be dragged or closed right away; Alt+Tab or a click in the client area captures as before. `TOTK_INPUT=none` disables the capture too.

`TOTK_STICK_DEADZONE` (0 to 50, default 8) is the gamepad stick dead zone in percent of full deflection, the same for both sticks. It is round: a stick inside the circle reads centred, and the rest of its travel is scaled to start at 0 at the edge, so full deflection still reaches full. Raise it for a stick that drifts. The game's own dead zone applies after it.

The stick direction bits (`StickLLeft` to `StickRDown`) are derived from the sticks at 50% deflection. A gamepad plugged in later is picked up within one second; with none connected the log says `input: no gamepad`.

## Nintendo Switch 2 controllers

Windows does not present the Switch 2 Pro Controller as a gamepad SDL can open (the log says `input: no gamepad`), so it needs a bridge program that wakes it and creates a virtual gamepad. The DS4Windows fork by hbashton ([github.com/hbashton/DS4Windows](https://github.com/hbashton/DS4Windows), tested with release VIIPERRC4.5.1, a pre-release) with the output set to an emulated DualSense gives buttons, rumble, gyro and accelerometer, and reads the sticks through the controller's calibration (they rest at 0 to 2%). The game reads its virtual DualSense directly; the log then shows `input: gamepad "DualSense Wireless Controller" (rumble, gyro and accelerometer, ...)` and `input: motion sensors on`. Its release notes also list Joy-Con 2 support, which is untested here.

Bridges that output a ViGEm DualShock 4 (ProCon2Windows, connectyourpro as tested) give buttons and rumble but no motion: their virtual DualShock 4 reports a gyro and an accelerometer that never change, so motion aiming and Ultrahand rotation do nothing. Run one bridge at a time; Steam Input can also claim the controller, so close Steam if the bridge does not find it.

## Scripted input

`TOTK_INPUT_SCRIPT=<file>` replays presses. It is ORed with real input and still applies under `TOTK_INPUT=none`. One event per line, `#` starts a comment:

```
<frame> <button> [hold_frames]            # A B X Y L R ZL ZR PLUS MINUS LEFT UP RIGHT DOWN LCLICK RCLICK; hold defaults to 4
<frame> LSTICK|RSTICK <x> <y> [hold_frames]   # -32767..32767, up is +y
@<ms> <button> [hold_ms]                  # timed on the world clock
@<ms> LSTICK|RSTICK <x> <y> [hold_ms]
[@<ms>|<frame>] wait world|file <name> [timeout_ms]   # hold the lines after it until the game gets there
```

`<frame>` counts 60 Hz ticks of game time since the first frame (presented frames at 60 fps), so the title menus get the presses at the same game time at any frame limit. `@<ms>` counts milliseconds of the world clock: game time stepped since the end of the first save load (TotK: the player leaves the title screen's origin; Bananza: the first frame), so a timed event lands at the same game time at any frame rate and load length. Both clocks add the game time each frame stepped (the real-time step, or whole vsyncs of the game's present interval without it), never the display time, so a load or a shader-compile stall does not move the events that follow. A later load, such as fast travel, still shifts what follows when the game presents a different number of frames during it. In a script with timed lines the untimed ones stop when the world clock starts, as the load ends at another tick in each run. Each event logs once with the time it asked for (`input: script frame N tick T (asked A): X`, `input: script frame N at T ms (asked A): X`), so two replays compare line by line; once no event can land any more a summary follows (`input: script done at frame N: L of M events landed, latest K ticks and D ms after the time asked`): fewer events than M, or a large lateness, mark a run whose input did not land as written. If the run ends first (the game quits, `--seconds` runs out), the exit path logs `input: script ended at frame N: L of M events landed, P pending` instead, so an early end shows how much of the script never ran. `tools/input/load_save.txt` loads the first save from the title screen.

Game time alone does not cover a load that runs in the background while the game keeps presenting frames, as the level load after the save-select screen does: on a slower machine it takes more game time, and a press timed for the end of it lands on the loading screen. A wait line holds every line after it until the game reaches a state the host can see, then shifts the clocks of those lines so they read the wait's time at that moment: the lines keep their spacing from the condition, earlier or later than in the recorded run. The conditions:

- `wait file <name>`: the game opens a file (RomFS or save) whose path contains `<name>`. Opens before the wait is reached do not count. A stream the game starts when a screen appears (the title music, the music of a level) marks that screen; `TOTK_TRACE=fs` logs each open with the world clock (`fs <s> @<ms> rom <path>`), which gives both the name and the time to write.
- `wait world`: the world clock has started (TotK: the first save load ended; Bananza: the first frame).

A wait is reached once the wait before it held (or at the start); the lines before it are not held back. Its time is optional (a bare `wait` is tick 0, so the lines after it count from the moment it held, and its timeout runs from the moment it was reached) and names the moment the condition held in the recorded run. A timed wait that holds before the world clock starts logs `held before the world clock` and does not shift the lines after it. If it has not held when its clock reaches its time plus `timeout_ms` (60000 by default, also for 0; always milliseconds of game time, and for a timed wait not counted before the world clock starts) it falls through as if it had held then. Each wait logs once: `input: script frame N: wait file X held at T ms (asked A): the lines after it move by +D ms` (or `timed out`); T, and the times logged for the lines after it, are on the script's clock, the world clock less the shifts of the earlier waits. The summary adds `input: script waits: W, K of them timed out`. For example, before the first press on the title screen and before the first press in the level (the times from the `TOTK_TRACE=fs` log of the recorded run):

```
@14567.0 wait file <title music stream>
@14952.2 A 133.3
...
@28367.2 wait file <level music stream>
@30171.8 PLUS 866.7
```

For A/B and image tests, `TOTK_FIXED_STEP=1` (or `DKB_FIXED_STEP=1`) with the real-time step on (`TOTK_REALTIME=1`) steps the game 1/60 s per presented frame whatever the frame took, so game time is frames / 60 in every run: the game runs slower than real time when frames take longer than 1/60 s. Without the real-time step the game already steps whole vsyncs of its own present interval, which its controller can raise when a frame runs late.

`TOTK_INPUT_RECORD=<file>` (`input_record` in `totk.ini`, or Input recording in the menu) writes the merged input in the same format, one line per button press and per stick segment: events before the world clock starts by tick, the rest timed; a button or stick held when the world clock starts is cut there and goes on as a timed event. The file replays as a script, which makes it the repro to attach to a bug report. The recording writes no wait lines: add them by hand where a press must follow a screen, with a `TOTK_TRACE=fs` log of the recorded run for the file and its time.

`TOTK_TEST_PAD=<lx>,<ly>,<rx>,<ry>` attaches an SDL virtual gamepad at start whose sticks rest at these raw SDL axis values (-32768 to 32767, down is +y), so a drifting or held stick can be tested without a controller: the values go through the gamepad mapping and `TOTK_STICK_DEADZONE` like a real pad's, and `TOTK_INPUT_RECORD` shows what reached the game.

## RenderDoc capture (blocked)

`TOTK_RDOC=<frame>[,...]` (`runtime/rdoc_win.c`; `TOTK_RDOC_DIR`, `TOTK_RDOC_DLL`) loads RenderDoc before `vkCreateInstance`. RenderDoc hides `VK_EXT_external_memory_host`, which the back end requires for guest pool imports (`runtime/vk_memory.c`), so device creation fails under it. A capture needs a copy-based pool fallback first.
