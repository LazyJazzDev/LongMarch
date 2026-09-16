# Sparkium on iOS (experimental)

Native SwiftUI app sharing Sparkium's JSON loader, materials, camera, accumulation,
film development and Metal ray-query pipeline with `sparkium_cli`. The scene picker
includes the original six demos and Blender Classroom, Junkshop and Monster.
The app runs full-screen in landscape. The film and camera aspect ratio follow
the full viewport, retaining the scene's camera pose and vertical field of view. A translucent left sidebar overlays the image,
following the desktop scene-control layout, and can be opened or dismissed without
resizing the viewport. Starting a render hides the sidebar; its corner button brings
it back, including while rendering. Controls remain native SwiftUI.
The existing JSON and scene assets are bundled unchanged, with the Sobol table.
Rendering uses one sample per dispatch on a serial background queue; Stop and
backgrounding request cancellation after the current operation. The render
resolution automatically uses the full viewport size in physical pixels
(SwiftUI size × display scale); the sidebar displays that resolution.
Large Blender scenes still load their original geometry and textures and may exceed
an iPhone's memory budget. Start with Cornell Box on a real device.

## Requirements

- Full Xcode with iOS SDK, macOS on Apple Silicon for resource preparation.
- iOS 18+ and a Metal device supporting ray queries and tier-2 argument buffers.
  This experiment requires ray queries; it does not substitute the software fallback.
  Simulator ray-query support depends on the runtime. Unsupported devices show an error.
- CMake 3.25+, Ninja, Python 3, the existing project DXC and SPIRV-Cross installation.
  DXC and SPIRV-Cross are used **only on the Mac**, and are not linked into the app.
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
open build-ios-device/SparkiumMobile.xcodeproj
```

For a device install, select the `Sparkium` target, choose your signing team and
connected iPhone/iPad in Xcode, and Run. The unsigned build is at
`build-ios-device/Release-iphoneos/Sparkium.app`.
For the simulator use a separate `build-ios-simulator` directory and
`-DCMAKE_OSX_SYSROOT=iphonesimulator`. The generated project supports both Debug
and Release; embedded HLSL uses the same release shader variant in both.

Resource preparation refuses to overwrite an existing output. Use `--output` to
prepare another bundle and `-DSPARKIUM_RESOURCES` to select it. Generated resources,
previews and build products remain outside Git; assets stay in the LFS submodule.

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
`SPARKIUM_SMOKE_SCENE=cornell_box`. The normal SwiftUI render flow runs at the native viewport resolution
and 2 spp, then saves `SmokeResult.json` and `SmokeResult.png` in the app's Documents
directory. For `simctl launch`, prefix this variable with `SIMCTL_CHILD_`.
This records unsupported-device errors as well as successful renders. Without the
variable the app waits for the user to select a scene and tap Render.

Automated bundle checks compare all nine replay images with the preparation images,
check a different resolution and 32-spp accumulation, and exercise missing/corrupt
cache errors. The MSL check compiles all cached stages for iOS 18 with fast math:

```sh
python3 platforms/ios/validate_bundle.py --renderer build-ios-replay/sparkium_mobile_check \
  --resources out/ios/Resources --output out/ios/validation
xcodebuild -downloadComponent MetalToolchain
python3 platforms/ios/validate_msl.py --resources out/ios/Resources --output out/ios/metal-validation
```
