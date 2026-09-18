# Sparkium CPU backend

Sparkium can trace paths on the host, with no graphics device involved. The
pipeline runs the same shader sources as the GPU backends and produces the same
images; this document covers how that works, how to run it, and what it does not
do yet.

## Running it

```sh
demo_sparkium_cli assets/scenes/cornell_box/scene.json \
  --backend host --pipeline cpu -o out.png
```

`--backend host` selects the host graphics backend, which backs buffers and
images with ordinary memory, and `--pipeline cpu` selects the CPU execution
path. Either the CLI's `--pipeline cpu` or a scene's `"renderer": {"pipeline":
"cpu"}` will do; the GUI offers both under **Pipeline → Path Tracing - CPU**.

`RENDER_PIPELINE_AUTO` never selects the CPU path. It is opt-in, because a CPU
render of a production scene takes minutes, not frames.

## How the shaders are shared

The CPU backend does not reimplement the renderer. The shaders in
`code/sparkium/shaders/` are compiled as C++ and executed natively:

- `common.hlsli`, the BSDFs, direct lighting, the shadow and subsurface code and
  the material samplers are used verbatim.
- `software/render.hlsl` — the same path tracer the compute fallback compiles
  with DXC — is included directly. Its `Main` delegates to `RenderDispatch`,
  which the CPU backend calls per pixel; the compute entry point stays for the
  GPU.
- `software/build.hlsl`'s five BVH passes became plain functions with the compute
  entry points kept behind `#ifndef SPARKIUM_CPU_SHADER`. The CPU backend builds
  its BVH differently (see below) but keeps the kernels usable.
- `tone_mapping.hlsl` gained a `ToneMapPixel` body so the film can be developed
  without a compute device.

Two things make this work:

**A compatibility layer.** `code/sparkium/pipelines/raytracing/cpu/shaders/`
provides the HLSL types the sources expect: packed `float2/3/4`, HLSL's
row-indexed matrices and `mul`, swizzles (including lvalue and permuting ones
such as `.wz`), the intrinsic set, and `ByteAddressBuffer`, `Texture2D`,
`ConstantBuffer` and `RWTexture2D` as views over host memory. glm cannot be used
for this: on ARM its vectors are 16-byte aligned, which would break the
buffer layouts the shaders reinterpret, and its swizzle operators only exist
under MSVC language extensions.

**A build-time rewrite.** `cmake/HlslAsCxx.cmake` copies the shader tree into the
build directory and rewrites the few constructs with no C++ spelling: `out` and
`inout` parameters become references, `in` disappears, `class` becomes `struct`
(HLSL classes have public members), `precise` is dropped, and `(SoftwareHit)0`
becomes aggregate initialisation. The checked-in shaders are never modified, and
the rewritten tree is regenerated when a shader changes.

Because the shared sources are the same text, the GPU and CPU paths cannot drift
apart. Where a divergence was unavoidable it was made explicit in the source:
the generated shader graphs broadcast through `SPARKIUM_SPLAT4`, which is
`(x).xxxx` on the GPU and a helper on the CPU.

## The host graphics backend

`grassland::graphics::BACKEND_API_HOST` backs `Buffer` and `Image` with
`std::vector<uint8_t>` and makes every GPU object — shader, program, command
context, acceleration structure, window — an inert stub. That is what lets the
front-end do its registration work unchanged: `GeometryMesh` uploads the same
byte layout, materials upload the same parameter buffers, and the CPU pipeline
reads them straight out of host memory.

## What the CPU pipeline does itself

`sparkium::raytracing::CpuPipeline` replaces the four things the GPU fallback
does with compute shaders:

- **Acceleration structures.** `bvh_builder.cpp` builds a binned SAH tree rather
  than replaying the GPU's Morton-and-bitonic-sort passes. Reusing the GPU
  algorithm would be slower to build and give a worse tree; what is shared is the
  node layout, so `software/traversal.hlsli` traverses it unchanged.
- **Light sampling.** `gather_light_power.hlsl` and the Blelloch scan become a
  sequential prefix sum, and `LightGeometryMaterial::SamplerPreprocessHost`
  reuses the compiled `MaterialEvaluator` shaders for per-primitive power.
- **Textures.** Registered images are converted once to RGBA floats, so sampling
  is free of format branches. The sampler is linear with repeat addressing, which
  is what the raytracing scene creates.
- **The film.** Accumulation lives on the host, so it survives across frames and
  `Film::Reset` clears it through the existing reset callback.

Pixel rows are split across `std::thread` workers. There is no thread pool in the
repository and one is not needed here: every frame spawns its workers and joins
them.

## Shader graphs

A shader graph's body is generated per scene at load time, so unlike every other
material it cannot be compiled ahead of time. The GPU backends hand the generated
source to DXC at runtime; the CPU backend evaluates it with one of two engines
behind `cpu::GraphProgram`:

- **Interpreter** (default). `graph_interpreter.cpp` lexes and parses the
  generated source and walks it. It evaluates the very text the GPU compiles, so
  it cannot disagree with it about what the graph means.
- **JIT.** `graph_jit.cpp` assembles the whole material set into one module and
  compiles it with Clang, linking LLVM and Clang into the backend. This needs a
  system LLVM; see below.

The JIT is **not enabled at runtime** yet. It assembles and compiles the
generated module correctly and every symbol it needs resolves, but calling into
the MCJIT'd code faults, and the cause has not been found. Until it is, the
backend withholds JIT programs and uses the interpreter, so `--graph-engine jit`
is accepted and falls back with a message rather than crashing.

To build it:

```sh
cmake -S . -B build -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
```

LLVM is deliberately not a vcpkg dependency: the port builds the whole compiler,
which takes hours, and the backend only wants it on desktop machines.
`LONGMARCH_ENABLE_CPU_JIT` can force it on or off.

## Verification

The CPU backend is checked three ways.

**Against the GPU.** `scripts/check_cpu_backend.py` renders every basic scene
through the compute fallback and on the CPU and compares the tone mapped PNGs.
It writes both renders, the CLI logs and `results.json` under `out/cpu-backend`:

```sh
python3 scripts/check_cpu_backend.py \
  --cli cmake-build-mac/demo/sparkium_cli/demo_sparkium_cli
```

| Scene | Mean absolute pixel difference (1 frame) |
| --- | --- |
| cornell_box | 0.001 / 255 |
| principled | 0.001 / 255 |
| specular | 0.001 / 255 |
| texture | 0.508 / 255 |
| blender_junkshop | 1.085 / 255 |
| blender_monster | 1.971 / 255 |
| point_light | 2.146 / 255 |
| blender_classroom | 2.828 / 255 |
| area_light | 5.350 / 255 |

Three scenes match to within 8-bit quantization. The rest differ because the two
backends build different acceleration structures, so rays that tie, or that a
light sampling CDF rounds differently, take different paths. On the light
sampling scenes that is Monte Carlo divergence and it falls with sample count:
area_light reads 5.35 at one frame, 0.38 at eight and 0.16 at thirty-two. On
textured and shader-graph scenes the difference plateaus at roughly 0.5 to 1.7
out of 255 and does not fall further, so a systematic component remains there;
see the [results report](cpu-backend-report.md) for the measurements and what
has been ruled out.

**With unit tests.** `test/sparkium/cpu_backend_test.cpp` renders a small
analytic scene with the host backend and checks that it produces light, that
accumulating over frames converges, that the film develops without a compute
device, and — where the device supports the compute fallback — that the two
backends agree to within 10%.

**Through the shared shaders.** `test/sparkium/cpu_bvh_test.cpp` builds a
binned SAH tree with the CPU builder, walks it with `InlineIntersect` from
`software/traversal.hlsli` compiled as C++, and compares every hit against a
double-precision brute force over all triangles. That covers the builder and the
node layout the two backends share in one test. It is a separate executable
because it includes the shader sources, which the backend's library defines too.

```sh
cmake --build <build> --target sparkium_cpu_bvh_test
./<build>/test/sparkium/sparkium_cpu_bvh_test
```

## Limitations

- Shader graphs are interpreted, which is slower per hit than compiled code.
  Scenes without graphs are unaffected.
- The BVH is rebuilt when the instance set changes, matching the compute
  fallback; geometry buffers are treated as immutable.
- Traversal uses the shader's fixed 32-entry stack, so the builder caps the tree
  depth.
- The host backend has no compute shaders, so pipelines other than the CPU one
  cannot run on it.
- Textured and shader-graph scenes keep a small systematic difference from the
  GPU (0.5 to 1.7 out of 255) that more samples do not remove.
