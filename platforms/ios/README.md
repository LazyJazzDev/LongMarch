# LongMarch Demos on iOS (experimental)

The app opens a native demo browser with **2048**, **Game of Life**, **Sparkium**,
**Graphics Hello** modules and **NBody CS**. Returning to Demos releases the active
renderer; backgrounding pauses rendering, simulation and game AI. Both portrait
and landscape are supported.

## Games

Both games compile their **original desktop UI and renderer** from `demo/2048`
and `demo/gol`: the same geometry, Slang lighting, font outlines, supersampled
resolve, layouts, button states and transitions. They render directly to Metal;
SwiftUI only supplies the native view container and navigation. The games have
separate C++ namespaces so both can live in one app without duplicate symbols.
`graphics::HostedWindow` supplies logical/drawable sizing and input events without
creating a GLFW window.

2048 uses its original Clear Sans vector font, palette, moving/merging tiles,
scoreboard, menu, new game and game-over transitions. Swipe to move; five quick
taps on the scoreboard enable the original AI worker, and one tap stops it.

Game of Life preserves the original shaded cells, pixel icons, concave/convex
size sliders, rotating blue die, red reset, pause/play spring and boundary-glider
animation. Tap or draw to edit, use two fingers to pan, and pinch to zoom. Its
width/height (2–256, initially 64×64), 1×/2×/5×/lightning speed, periodic/fixed edges and file buttons
are the same desktop controls. Lightning advances 60 generations per second.
Sidebars follow the window's short edges rather than the grid aspect ratio. On
iPhone, GoL fills the display, with reset/random at the two upper corners in
portrait, dense controls below, and a compact status/exit overlay beneath the
camera safe area. GoL locks the entire interface to its entry orientation. Device direction
notifications rotate only the button icons with a short animation; the page,
button positions, sliders, grid and touch coordinates stay fixed, with no system
window-rotation animation. Flat/unknown directions preserve the last icon angle.
Leaving GoL restores normal interface rotation for the browser and other demos.
GoL defers bottom-edge system gestures and keeps the lower controls above the
window's home-indicator safe area plus 12 points. The canvas still fills the
screen. This reduces accidental Home gestures; iOS still allows users to leave
the app and does not permit ordinary apps to disable Home navigation entirely.
On iOS, tap either W/H readout to open a system popover with a native 2–256 slider
and single-cell stepper. Scrubbing updates the displayed value; release applies
the new grid size once. Done or tapping outside closes the popover. The compact
readouts do not change values when dragged; desktop sliders retain direct drag.
Original open/save buttons launch native Files pickers; import retains larger
axes and centers the pattern independently per dimension. Export saves the full
grid, including dead borders. Returning from the background resets frame clocks
so inactive time is not simulated or applied to animations.

## Graphics Hello

Triangle, Texture, Blend, Cube, Resize, SDR Sample, HDR and Ray Query reuse the
current `demo/graphics_hello/modules` Slang. Ray Query displays the rotating triangle
and procedural sphere and requires device ray-query support. The HDR module uses
a floating-point EDR Metal drawable, a 0–3 linear gradient, reference white and an
HDR/SDR toggle. Unsupported full RT pipeline modules (Ray Tracing, External Shader,
RT Multi Shader Group) remain visible with their Metal limitation explained.

MTKView presents GPU textures without a per-frame CPU download. Resize and NBody
follow drawable size with a render scale control; other modules retain 1280 × 720,
fitted to the screen. NBody includes particle/galaxy count, time step, pause, reset
and drag-to-rotate controls. It defaults to 4096 particles and supports up to 65536.
Frame time, GPU time and throughput are displayed.

## Sparkium

Native SwiftUI app sharing Sparkium's JSON loader, materials, camera, accumulation,
film development and Metal ray-query pipeline with `sparkium_cli`. The scene picker
includes the original six demos and Blender Classroom, Junkshop and Monster.
The Sparkium demo runs full-screen in landscape and renders at the resolution and camera
aspect ratio stored in each scene JSON. The image initially fits the screen.
Pinch to zoom, drag to pan, and double-tap to fit again. Progressive image updates
preserve the viewing transform; gestures never change the render camera.

HDR is enabled by default. Film development produces linear sRGB floating-point
pixels (including values above 1), presented directly as CGImage contents on a
Core Animation layer requesting high dynamic range (without UIImage conversion).
Images carry their exposed RGB content headroom so iOS can tone-map
highlights to the available screen range instead of clipping them. The actual brightness depends on the device’s available EDR headroom.
The HDR toggle switches to the scene’s SDR view transform; exposure adjustments
redevelop the existing film, even after reaching the sample limit or pausing,
without clearing or adding samples. Pinch/pan transforms survive display changes.
The CPU image preview is 32-bit float RGBA for HDR and 8-bit RGBA for SDR.

A translucent left sidebar follows the desktop control layout using native UI.
The initial scene and every newly selected scene start rendering automatically.
The spp picker controls only the accumulation limit: raising it continues the
existing film; lowering it preserves accumulated samples and finishes any sample
already in flight. Reload loads the scene again; Reset film clears accumulation
without reloading scene resources. Backgrounding pauses after the current sample,
and returning resumes toward the current limit.

The sidebar reports camera Ray/s and FPS for the last rendered frame, accumulated
spp, render time, backend, device, pipeline, resolution, samples per frame and
maximum bounces. Ray/s is width × height × samples per frame × FPS, matching the
desktop camera-ray estimate. Blender scene names are Blender Classroom, Blender
Junkshop and Blender Monster. Asset JSON files remain unchanged.

The application icon is the bundled Cornell Box rendered at 1024 × 1024 and
1024 spp through the Metal ray-query pipeline. Its opaque PNG is stored in
`Assets.xcassets/AppIcon.appiconset`; Xcode generates the iPhone and iPad icon sizes
when building the application.

The iOS packager limits PNG/JPEG textures to a 1024-pixel longest edge by default,
preserving aspect ratio and PNG alpha. Smaller textures are copied unchanged.
This reduces decoded texture memory, especially for Blender Junkshop; it is
texture downsampling, not ASTC GPU compression. Source assets, JSON references,
geometry and scene render resolutions remain unchanged. HDR/EXR files are copied
unchanged. Large scenes can still exceed a device's memory budget.

## Requirements

- Full Xcode with iOS SDK, macOS on Apple Silicon for resource preparation.
- iOS 18+ and a Metal device supporting tier-2 argument buffers and unified memory.
  Sparkium additionally requires ray queries; graphics/NBody do not. The simulator supports the raster/compute game path; ray-query support is checked separately.
- CMake 3.25+, Ninja, Python 3, and the shared mobile Slang and SPIRV-Cross packages below.
  Slang and SPIRV-Cross are used **only on the Mac**, and are not linked into the app.
- Pillow for resizing textures during packaging: `python3 -m pip install -r platforms/ios/requirements.txt`.
- Material edits or shader edits require regenerating the resource bundle.

## Build

Run from the repository root. Initialize `assets` and fetch its LFS objects first.
Install the shared host dependencies once (Slang, SPIRV-Cross, metal-cpp,
portable headers, and FreeType/MikkTSpace sources):

```sh
/path/to/vcpkg/vcpkg install --x-manifest-root=platforms/mobile \
  --x-install-root=out/mobile/deps --triplet=arm64-osx
```

Both mobile projects discover this installation automatically. CMake never downloads
SDKs or third-party sources. FreeType and MikkTSpace are compiled with the target
SDK; no macOS library is linked into an iOS/HarmonyOS app. `LONGMARCH_MOBILE_DEPS`
can select a nondefault installation; normal builds need no dependency path flags.
The Slang library requires macOS 26+ on Apple Silicon. An external Slang SDK can be
selected with `slang_DIR`; it must meet the desktop minimum version. Regenerate the
bundle after changing the Slang compiler.

```sh
cmake -S platforms/ios -B build-ios-prepare -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSPARKIUM_PREPARE=ON -DSPARKIUM_APP=OFF
cmake --build build-ios-prepare
python3 platforms/ios/prepare_resources.py \
  --renderer build-ios-prepare/sparkium_mobile_check

cmake -S platforms/ios -B build-ios-device-ninja -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0
cmake --build build-ios-device-ninja
```

Ninja builds an unsigned `build-ios-device-ninja/LongMarch.app`, including the
asset catalog and prepared resources. To build for Simulator use a separate
`build-ios-simulator-ninja` directory and `-DCMAKE_OSX_SYSROOT=iphonesimulator`.
Install on a booted simulator with:

```sh
codesign --force --sign - build-ios-simulator-ninja/LongMarch.app
xcrun simctl install booted build-ios-simulator-ninja/LongMarch.app
xcrun simctl launch booted dev.lazyjazz.longmarch
```

For a signed device install, the existing Xcode workflow is also supported: use
`-G Xcode` in a **separate** directory, open `LongMarch.xcodeproj`, select your
Apple development team and connected device, and Run. The bundle identifier is
`dev.lazyjazz.longmarch`; the display name is **LongMarch Demos**.

### Game of Life app

The same project also builds **Game of Life** (生命游戏, `dev.lazyjazz.gameoflife`), a
standalone app with only the GoL page: no demo browser and no status header. It
bundles the pattern library and the few shaders the game reads, about 200 KB
instead of the full scene bundle. Extract them from a prepared bundle with a replay
build (`SPARKIUM_PREPARE=OFF`) of `mobile_demo_check`. The `GameOfLife` target is
added whenever `GAME_OF_LIFE_RESOURCES` (default `out/ios/GameOfLifeResources`)
exists:

```sh
python3 platforms/ios/prepare_game_resources.py --source out/ios/Resources \
  --checker build-ios-replay/mobile_demo_check --output out/ios/GameOfLifeResources
```

A signed device build uses the Xcode generator, and automatic signing creates the
provisioning profile:

```sh
cmake -S platforms/ios -B build-ios-xcode -G Xcode -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0 -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=<team>
xcodebuild -project build-ios-xcode/LongMarch.xcodeproj -scheme GameOfLife \
  -configuration Release -destination id=<device> -allowProvisioningUpdates build
xcrun devicectl device install app --device <device> \
  build-ios-xcode/Release-iphoneos/GameOfLife.app
```

`demo/gol/tools/app_icon.py` generates its Icon Composer icon (`gol/AppIcon.icon`)
from the boundary-mode button: the plate gradient as the fill and an enlarged glider
as a Liquid Glass layer. actool renders the flat icons of earlier iOS versions from
it. Preview a rendition with Icon Composer's `ictool`, for example
`ictool gol/AppIcon.icon --export-image --output-file icon.png --platform iOS
--rendition Dark --width 1024 --height 1024 --scale 1`.

Resource preparation refuses to overwrite an existing output. Use `--output` to
prepare another bundle and `-DSPARKIUM_RESOURCES` to select it. Generated resources,
previews and build products remain outside Git; assets stay in the LFS submodule.
Use `--texture-max-dimension 512` for a smaller texture budget, or `0` to preserve
original texture files. `texture-report.json` records original/packaged dimensions
and estimated RGBA8 storage per texture. `asset-hashes.json` hashes the actual
packaged files, and shader preparation renders those same files.

## Shader architecture

The Mac preparation tool loads and renders each bundled scene through the same
`RenderSession` used by the app. It records Slang → SPIR-V and SPIR-V → iOS MSL
(including entry point, argument-buffer slots and thread-group size). Scene entity
registration follows insertion order so generated material code and bindings are
stable across processes and platforms.

The mobile build embeds the Slang VFS to reproduce shader request keys but never
runs Slang or SPIRV-Cross. It loads the recorded shader and compiles its MSL using
Metal's runtime compiler. This retains first-use Metal compilation cost. Shader
keys use a Slang-specific, versioned namespace (old DXC bundles must be regenerated)
and include all VFS file contents and compiler arguments; MSL keys also include
SPIR-V, platform and resource bindings. Cache files carry SHA-256 integrity hashes.
Missing/stale/corrupt entries produce a visible error. The app never writes into
its signed resource bundle.

## Validation

Check texture packaging independently with:

```sh
python3 -m unittest discover -s platforms/ios/tests -p 'test_texture_assets.py'
```

Build a macOS replay executable with `SPARKIUM_PREPARE=OFF` and `SPARKIUM_APP=OFF`,
then render from the prepared bundle. This executable has the same no-Slang/no-SPIRV-Cross
configuration as iOS:

```sh
cmake -S platforms/ios -B build-ios-replay -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSPARKIUM_APP=OFF
cmake --build build-ios-replay
build-ios-replay/sparkium_mobile_check out/ios/Resources cornell_box out/ios/cornell.png replay 256 32
```

For automated simulator/device smoke runs, launch with environment variable
`SPARKIUM_SMOKE_SCENE=cornell_box`. The normal SwiftUI render flow runs at the scene's preset resolution
and 2 spp, then saves `SmokeResult.json` and `SmokeResult.png` in the app's Documents
directory. For `simctl launch`, prefix this variable with `SIMCTL_CHILD_`.
This records unsupported-device errors as well as successful renders. Without the
variable the application opens the demo list. Entering Sparkium automatically
starts its initial scene with the normal sample limit.

Set `LONGMARCH_SMOKE_DEMO=nbody_cs` (or one of the graphics module identifiers) to
open that demo directly. After three frames it writes `DemoSmokeResult.json` to
Documents, including device, render dimensions and frame/GPU times.

Run shared game adapter checks with `build-ios-replay/mobile_games_check out/ios/Resources`.
Each scene replay also checks finite HDR output, exposure scaling and retained
sample counts while changing display settings.

The preparation build also produces `mobile_demo_check`. Resource preparation
uses it to cache the graphics and compute shaders. Verify all eleven runnable demos using:

```sh
build-ios-replay/mobile_demo_check out/ios/Resources all out/ios/demos-replay
```

This runs actual raster/compute work, checks finite pixels and particle positions,
and verifies NBody motion, pause and fresh particle distributions on consecutive resets,
plus render-target resize. Like the desktop demo, NBody seeds its random generator
once at startup and advances the sequence across resets.

Automated bundle checks compare all nine replay images with the preparation images,
check a different resolution and 32-spp accumulation, and exercise missing/corrupt
cache errors. The MSL check compiles all cached stages for iOS 18 with fast math:

```sh
python3 platforms/ios/validate_bundle.py --renderer build-ios-replay/sparkium_mobile_check \
  --resources out/ios/Resources --output out/ios/validation
xcodebuild -downloadComponent MetalToolchain
python3 platforms/ios/validate_msl.py --resources out/ios/Resources --output out/ios/metal-validation
```

The render-controller check exercises preset resolutions, retaining accumulation
when changing limits, pause/resume, rapid film resets and superseded scene loads:

```sh
cmake -S platforms/ios -B build-ios-replay -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-ios-replay
python3 platforms/ios/tests/check_render_controller.py \
  --build build-ios-replay --resources out/ios/Resources
```

NBody uses a linear RGBA16Float EDR presentation surface and the desktop
HDR particle brightness conversion. Its HDR switch preserves simulation state.

### Game rendering power usage

The shared game UI exposes its next frame deadline. The iOS host stops the
MTKView display loop while a board is still, wakes on input/resize/focus/file
completion, and schedules normal Life generations at their simulation deadlines.
Animations and lightning mode use the native 60 Hz display link;
only slower simulation deadlines use one-shot timers; lightning advances
60 generations per second, one per 60 Hz frame. Hosted surfaces render at native resolution without
supersampling, since phone displays are dense enough, and allocate the second full-screen
color target only when an overlay transition uses it. Frames without overlays bypass the
resolve pass; the presentation command uses the same ordered Metal
queue without a redundant wait between game rendering and presentation.
Presentation completion is asynchronous with at most two game frames in flight;
Metal queue ordering protects shared textures and completion handlers report errors.

`mobile_games_check` checks idle/animation transitions and Life deadlines.
For device idle/wake checks, launch a game with `LONGMARCH_SMOKE_DEMO=gol` (or
`2048`) and `LONGMARCH_SMOKE_IDLE=1`. `IdleSmoke0.json` and `IdleSmoke1.json` in
Documents record frame counts across two idle seconds before and after injected
input. These are rendering-work checks, not battery-life measurements.

Large Life grids use a four-vertex cell quad with the original eighth-power
rounded contour evaluated in the fragment shader, replacing 120 triangles per
cell. Fully clipped cells are culled on the CPU and placement matrices are
cached across frames. Settled cell appearance is cached as well, avoiding
repeated interpolation for unchanged cells. Only the small list of contiguous drawing batches is
sorted; the 40,000 instance records retain their original storage order. Input dispatch uses contiguous listener snapshots and
only rechecks membership if a callback changes the listener set.

`gol_benchmark <resources>` compares 40²/200² grids at 1290×2409, measuring
CPU submission, GPU-completed frame time and pointer dispatch. On iPhone, launch
with `LONGMARCH_SMOKE_DEMO=gol`, `LONGMARCH_SMOKE_GOL_SIZE=200` and
`LONGMARCH_SMOKE_BENCHMARK=1` to collect 60 frames after 10 warm-up frames in
Documents/BenchmarkResult.json. After frame 70 the benchmark resumes normal simulation scheduling. Set
`LONGMARCH_SMOKE_AUTORUN=1` to record frame counts over two input-free seconds
in Documents/AutorunSmoke.json and verify timer-driven updates.
`LONGMARCH_SMOKE_ORIENTATION=left|right|portrait` supplies a device direction
and verifies that an actual scene rotation request is rejected while bounds and
view transforms stay fixed, recorded in Documents/OrientationSmoke.json. Use
`exit` to additionally verify rotation is restored after releasing the lock
(Documents/OrientationExitSmoke.json). `mobile_games_check` also verifies that rotating
icons leaves grid pixels unchanged and returns the paused renderer to idle.

Device benchmark FPS uses wall-clock elapsed time; submission timings exclude
asynchronous presentation completion.

The serial native render queue uses foreground interactive QoS for visible
frames; idle games still schedule no work.

`LONGMARCH_SMOKE_SIZE_PICKER=width|height` taps the corresponding portrait readout
through the normal native input path for popover screenshots. `mobile_games_check`
verifies tap-only activation, drag cancellation, range limits and selected values.
For store screenshots, `LONGMARCH_SMOKE_TAPS="x,y;x,y"` taps points given as fractions
of the game view one second apart, and `LONGMARCH_SMOKE_PATTERN=<name>` makes the open
button load that built-in pattern (for example `Gosper glider gun`) instead of
presenting the library.
