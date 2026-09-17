# LongMarch Demos on iOS (experimental)

The app opens a native demo list. Choose Sparkium to access its scene picker, or
run Hello Triangle, Hello Texture, Hello Blend, Hello Resize, Hello SDR Sample,
and NBody CS independently. The Demos button returns to the list and releases the
current demo's GPU resources. Backgrounding pauses rendering and simulation.

Graphics demos reuse their desktop HLSL, vertex data and LongMarch graphics API.
An MTKView presents the rendered Metal texture directly, without per-frame CPU
image downloads. Resize and NBody follow the drawable size and offer a render
scale control; the other graphics tests retain their desktop 1280 × 720 target,
fitted to the screen. NBody includes particle count, galaxy count, time step,
pause/resume, reset and drag-to-rotate controls. It defaults to 4096 particles;
65536 remains selectable. Gravity is normalized by the selected count so total
mass remains constant. Frame render time, GPU time and throughput are displayed.

Hello Ray Tracing and RT Multi Shader Group are listed as unavailable because
the current Metal backend does not implement full RT pipelines, procedural AABB
intersections or callable shaders. Sparkium's Metal Ray Query remains available.

## Sparkium

Native SwiftUI app sharing Sparkium's JSON loader, materials, camera, accumulation,
film development and Metal ray-query pipeline with `sparkium_cli`. The scene picker
includes the original six demos and Blender Classroom, Junkshop and Monster.
The Sparkium demo runs full-screen in landscape and renders at the resolution and camera
aspect ratio stored in each scene JSON. The image initially fits the screen.
Pinch to zoom, drag to pan, and double-tap to fit again. Progressive image updates
preserve the viewing transform; gestures never change the render camera.

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
  Sparkium additionally requires ray queries; graphics/NBody do not. The current
  simulator may lack the required Metal capabilities; unsupported devices show an error.
- CMake 3.25+, Ninja, Python 3, the existing project DXC and SPIRV-Cross installation.
  DXC and SPIRV-Cross are used **only on the Mac**, and are not linked into the app.
- Pillow for resizing textures during packaging: `python3 -m pip install -r platforms/ios/requirements.txt`.
- Material edits or shader edits require regenerating the resource bundle.

## Build

Run from the repository root. Initialize `assets` and fetch its LFS objects first.
Obtain the portable dependency headers using the small manifest in this directory:

```sh
/path/to/vcpkg/vcpkg install --x-manifest-root=platforms/ios \
  --x-install-root=out/ios/deps --triplet=arm64-osx
```

You can also reuse an existing LongMarch vcpkg include directory. No host dependency
libraries are linked: fmt is header-only and MikkTSpace is compiled from its pinned
source. CMake fetches the pinned official metal-cpp headers.

```sh
cmake -S platforms/ios -B build-ios-prepare -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSPARKIUM_PREPARE=ON -DSPARKIUM_APP=OFF \
  -DSPARKIUM_HEADERS="$PWD/out/ios/deps/arm64-osx/include"
cmake --build build-ios-prepare
python3 platforms/ios/prepare_resources.py \
  --renderer build-ios-prepare/sparkium_mobile_check

cmake -S platforms/ios -B build-ios-device -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0 \
  -DSPARKIUM_HEADERS="$PWD/out/ios/deps/arm64-osx/include"
cmake --build build-ios-device --config Release -- CODE_SIGNING_ALLOWED=NO
open build-ios-device/LongMarch.xcodeproj
```

For a device install, select the `LongMarch` target, choose your signing team and
connected iPhone/iPad in Xcode, and Run. The unsigned build is at
`build-ios-device/Release-iphoneos/LongMarch.app`.
The Xcode project and application target are `LongMarch`; the bundle identifier
is `dev.lazyjazz.longmarch`. The installed display name remains **LongMarch Demos**.
Select your existing Apple development team for automatic signing. The new bundle
identifier installs separately from the earlier `dev.lazyjazz.sparkium` app.
For the simulator use a separate `build-ios-simulator` directory and
`-DCMAKE_OSX_SYSROOT=iphonesimulator`. The generated project supports both Debug
and Release; embedded HLSL uses the same release shader variant in both.

Resource preparation refuses to overwrite an existing output. Use `--output` to
prepare another bundle and `-DSPARKIUM_RESOURCES` to select it. Generated resources,
previews and build products remain outside Git; assets stay in the LFS submodule.
Use `--texture-max-dimension 512` for a smaller texture budget, or `0` to preserve
original texture files. `texture-report.json` records original/packaged dimensions
and estimated RGBA8 storage per texture. `asset-hashes.json` hashes the actual
packaged files, and shader preparation renders those same files.

## Shader architecture

The Mac preparation tool loads and renders each bundled scene through the same
`RenderSession` used by the app. It records HLSL → SPIR-V and SPIR-V → iOS MSL
(including entry point, argument-buffer slots and thread-group size). Scene entity
registration follows insertion order so generated material code and bindings are
stable across processes and platforms.

The mobile build embeds the HLSL VFS to reproduce shader request keys but never
runs DXC or SPIRV-Cross. It loads the recorded shader and compiles its MSL using
Metal's runtime compiler. This retains first-use Metal compilation cost. Shader
keys include all VFS file contents and compiler arguments; MSL keys also include
SPIR-V, platform and resource bindings. Cache files carry SHA-256 integrity hashes.
Missing/stale/corrupt entries produce a visible error. The app never writes into
its signed resource bundle.

## Validation

Check texture packaging independently with:

```sh
python3 -m unittest discover -s platforms/ios/tests -p 'test_texture_assets.py'
```

Build a macOS replay executable with `SPARKIUM_PREPARE=OFF` and `SPARKIUM_APP=OFF`,
then render from the prepared bundle. This executable has the same no-DXC/no-SPIRV-Cross
configuration as iOS:

```sh
cmake -S platforms/ios -B build-ios-replay -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSPARKIUM_APP=OFF -DSPARKIUM_HEADERS="$PWD/out/ios/deps/arm64-osx/include"
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

Set `LONGMARCH_SMOKE_DEMO=nbody_cs` (or one of the graphics directory names) to
open that demo directly. After three frames it writes `DemoSmokeResult.json` to
Documents, including device, render dimensions and frame/GPU times.

The preparation build also produces `mobile_demo_check`. Resource preparation
uses it to cache the graphics and compute shaders. Verify all six demos using:

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
