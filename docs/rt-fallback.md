# Sparkium compute ray tracing fallback

The `rt-fallback` branch is based on `blender-align` and keeps its JSON scenes,
Blender material graphs and triangulated hair. Its `assets` submodule points to
the `blender-align` snapshot in `LazyJazzDev/LongMarchAssetsLFS`; the old assets
repository and its history are unchanged.

## Selecting the backend

| Pipeline | Behavior |
| --- | --- |
| `auto` | Hardware ray tracing when available, otherwise compute ray tracing |
| `ray_tracing` | Hardware ray tracing when available, otherwise compute ray tracing |
| `rt_fallback` | Force compute ray tracing, including on hardware RT devices |
| `rasterization` | Existing raster renderer |

Use `renderer.pipeline` in scene JSON, the GUI's **Compute ray tracing** option,
or `sparkium::RENDER_PIPELINE_RT_FALLBACK` in C++. The CLI accepts:

```sh
cmake-build-rt-fallback/demo/sparkium_cli/demo_sparkium_cli \
  assets/scenes/cornell_box/scene.json --pipeline rt_fallback -o output.png
```

For hardware comparisons, always add `--require-hardware-rt` to a
`--pipeline ray_tracing` run. It fails if the selected device lacks hardware RT,
preventing accidental comparison of the fallback against itself. `--debug`
enables graphics validation.

The fallback requires the engine's existing compute, descriptor-array and
storage-image support. It does not create native acceleration structures or
dispatch ray tracing commands. On non-RT devices it also skips RT shader
libraries. Hardware-capable devices may still compile those libraries while
constructing scene components, even when compute rendering is selected.

## Implementation

`SoftwarePipeline` builds BLAS and TLAS in ordinary GPU buffers using compute
shaders. Leaf bounds are reduced, assigned Morton keys, sorted by a GPU bitonic
sort, and reduced again into a balanced binary tree. CPU work allocates buffers,
registers instances and records dispatches; bounds construction and sorting run
on the GPU. Nodes occupy 32 bytes and use heap indexing with power-of-two padding.

BLAS are cached for unchanged geometry identities and primitive counts. TLAS
are rebuilt when rendering to reflect current transforms and active instances.
Geometry buffers are assumed immutable, matching the existing mesh API; in-place
deformation needs explicit cache invalidation before it can be supported.

Traversal uses object-space BLAS and world-space TLAS, separate fixed-size
stacks, parallel-safe slab tests and a watertight triangle test. Transformed ray
directions retain their length so that intersection distances remain correct
under nonuniform and mirrored scales. Transparent shadows accumulate material
transmission across intersections; opaque shadows stop at the first hit.

The two tracing backends share hit-record construction, camera/lens sampling,
the path loop, BSDFs, direct lighting, shader graph code, subsurface random walks,
film accumulation and tone mapping. Shader graph samplers are compiled into
namespaces in the compute shader. The existing hair representation consists of
triangle tubes and uses the same mesh path. This is a compute path tracer, so it
retains indirect lighting and reflections from the RT renderer.

## Validation

Build using the repository's normal CMake/vcpkg configuration, then:

```sh
cmake --build cmake-build-rt-fallback \
  --target sparkium_fallback_test demo_sparkium_cli demo_sparkium_gui -j 6
ctest --test-dir cmake-build-rt-fallback/test -R sparkium_fallback --output-on-failure
python3 scripts/check_rt_fallback.py \
  --cli cmake-build-rt-fallback/demo/sparkium_cli/demo_sparkium_cli \
  --graph-smoke --debug
```

The snapshot script requires Pillow and writes scene copies with absolute asset
paths, images, logs, a contact sheet and `results.json` under `out/rt-fallback`.
Its default scenes are Cornell box, area light, point light, principled,
specular and texture. `--graph-smoke` adds real Blender Monster material graphs
to Cornell geometry, plus opacity and random-walk subsurface parameters.

For comparisons on an RT device:

```sh
python3 scripts/check_rt_fallback.py \
  --cli cmake-build-rt-fallback/demo/sparkium_cli/demo_sparkium_cli \
  --graph-smoke --compare-hardware --spp 256 --output out/rt-comparison
```

The script reports normalized RGB MAE/RMSE of tone-mapped PNGs. An optional
`--max-rmse` sets a failure threshold. These metrics complement visual inspection;
they do not establish HDR equivalence. The C++ parity test compares raw HDR
output for a small emissive scene and skips explicitly without RT hardware.

Verified on Apple M5 / MoltenVK 1.4.1:

- GPU intersections against an independent double-precision CPU oracle: 1, 5
  and 257 triangles, 1,030 rays per configuration, four instance-update phases
  (12,360 comparisons), including degeneracy, duplicate centroids and scaling.
- Empty-scene background, accumulation/reset, 17×13 output, and two transparent
  shadow layers with expected transmission of 0.25.
- Hardware RT libraries and five compute BVH kernels compile to SPIR-V and DXIL.
- All six basic scenes and the graph smoke render at 96×96, 32 spp, 12 bounces,
  with graphics validation enabled and no reported validation errors.

Hardware image parity remains unverified locally because the device exposes no
hardware RT. D3D12 runtime behavior also needs verification on Windows. Snapshot
wall times include scene loading and shader compilation and are not benchmarks.

## Current limits

- This is an experimental fallback. BVH build cost, divergent traversal and the
  combined material shader have not been optimized for large production scenes.
- Each tree accepts at most 2,097,152 leaves to stay within baseline dispatch
  dimensions; total node byte addresses must fit in 32 bits. Singular instance
  transforms are rejected. Procedural/custom geometry needs a traversal adapter.
- Descriptor limits still apply. Full Blender Monster exceeds the regular
  storage-buffer binding limit of 31 on the tested MoltenVK configuration. Merely
  setting `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=1` did not remove this limit.
  Resource packing or an update-after-bind descriptor path is future work.
- Coincident transparent surfaces and edge ties can resolve differently between
  software and native traversal. Monte Carlo path divergence prevents a general
  promise of bit-identical output.

The classroom lamp meshes use distinct names on case-insensitive filesystems:
`blackboard_lamp_fixture_m0.spmesh` for the fixture and
`blackboard_lamp_emitter_m0.spmesh` for the emitting surface. Their original
binary contents are preserved, and scene geometry IDs and references match
these names.

## Frame profiling

Use `--frames 24 --profile timings.csv` with the CLI for optional Vulkan GPU
timestamps and host wall-time scopes. `scripts/profile_rt_fallback.py` runs the
six basic scenes with warmup exclusion and produces CSV / JSON results.
See [the M5 performance analysis](rt-fallback-profile.md) for measurements,
profiling overhead checks, and the distinction between GPU work and host waits.

## Native Apple Silicon backend

Sparkium also runs the compute fallback and raster pipeline through metal-cpp.
Use `--backend metal`; tier 2 argument buffers remove the 31 direct-buffer-slot
constraint of the tested MoltenVK path. See [Metal backend details](metal-backend.md)
for builds, comparisons, diagnostics, and limitations.
