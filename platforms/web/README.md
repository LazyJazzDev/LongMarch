# LongMarch web apps

2048, Game of Life and N-body run in the browser on WebGPU. The engine and the
desktop game UIs compile to WebAssembly with Emscripten, draw through the
WebGPU backend (`code/grassland/graphics/backend/webgpu`) and reuse the mobile
adapters (`platforms/ios/demos/DemoSession.cpp`, `games/DesktopGameSession.cpp`).
One page, `index.html`, selects the app with `?app=gol`, `?app=2048` or
`?app=nbody`; the build is a static site.

The browser compiles no Slang. Like the mobile apps, shaders come from a cache
prepared on the build host: `web_shader_prepare` repeats each shader request of
the three apps with the WebGPU backend's `-target wgsl` arguments, so the cache
keys match.

## Build

Build the preparation tool in a macOS `SPARKIUM_PREPARE` build of
`platforms/ios` (see `platforms/ios/README.md`), then stage the resources:

```sh
cmake --build build-ios-prepare --target web_shader_prepare
python3 platforms/web/prepare_resources.py --shader-tool build-ios-prepare/web_shader_prepare
```

Regenerate them after changing a shader of the three apps. Then build with an
activated [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)
(4.0 or later, for the `emdawnwebgpu` port):

```sh
emcmake cmake -S platforms/web -B build-web -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-web
python3 -m http.server --directory build-web 8765
```

Open http://localhost:8765/. `build-web` holds the site: `index.html`,
`longmarch.js`, `longmarch.wasm` and `longmarch.data` (the prepared resources).

## Browser support

WebGPU is required: current Chrome and Edge on desktop and Android, and Safari 26.
The page reports when it is unavailable.

## Differences from the native apps

- Browser pages cannot wait for the GPU. `WaitGPU` returns immediately (WebGPU
  keeps objects alive while queued work uses them), and GPU readback is
  unavailable, so screenshots through `DownloadData` are not supported.
- Static hosting cannot enable the cross-origin isolation that threads need.
  2048's autoplay searches on the main thread with a 50 ms budget instead of a
  worker thread.
- N-body renders to a half-float target: WebGPU blends 32-bit float targets only
  with an optional feature.
- Game of Life's pattern library and file dialogs are not available yet; its
  open and save buttons do nothing.
- Input: mouse and one-finger touch act as the pointer, the wheel scrolls, a
  trackpad pinch zooms, and the keyboard works as on the desktop.
