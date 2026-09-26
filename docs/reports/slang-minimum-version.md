# Minimum supported Slang compiler

Tested on macOS ARM64 on 2026-09-26. The compiler requirement is **2026.18.1 or
newer**, not exactly 2026.18.3. The upstream vcpkg port currently supplies
2026.18.2; an external SDK meeting the minimum is equally valid.

## Evidence and scope

The earliest tested release with the storage-buffer resource-operand fix is
**2026.11**; its immediately preceding release 2026.10.2 fails those cases.
Upstream [#10656](https://github.com/shader-slang/slang/pull/10656) merged on
2026-06-09, between these releases, and adds the missing resource decorations
and capabilities. This is sufficient for the original direct/local-alias cases.

However, arithmetic after `NonUniformResourceIndex` still loses the annotation
in 2026.18. Upstream [#13089](https://github.com/shader-slang/slang/pull/13089)
fixes that path; the next release **2026.18.1** passes all 12 additional integer
composition cases at both optimization levels. We require that stronger floor
rather than accepting a compiler known to drop marks in ordinary index math.

The test includes the previous 17-case corpus plus add, subtract, multiply,
divide, modulo, left/right shifts, bitwise AND/OR/XOR/NOT and unary negation after
the annotation. Each is compiled at O0/O3: 58 outputs/version, 348 total.
We inspect actual buffer-load pointers for NonUniform and storage-buffer
nonuniform capability, and run `spirv-val --target-env vulkan1.2`. No SPIR-V
patching is performed. All 348 modules pass validation; validation alone does
not establish correct nonuniform semantics.

| Release | Original supported cases (28) | Arithmetic cases (24) | Known caller-only failures (4) | Uniform controls (2) |
| --- | ---: | ---: | ---: | ---: |
| 2026.10.2 | 0 | 0 | 4 | 2 |
| 2026.11 | 28 | 0 | 4 | 2 |
| 2026.18 | 28 | 0 | 4 | 2 |
| 2026.18.1 | 28 | 24 | 4 | 2 |
| 2026.18.2 | 28 | 24 | 4 | 2 |
| 2026.18.3 | 28 | 24 | 4 | 2 |

There is **no tested version that makes every possible annotation placement
correct**. In all six versions, annotating only an integer argument in the
caller does not reliably propagate into user-defined callees (with or without
`noinline`). Annotate at the actual descriptor access. Select/phi and arbitrary
interprocedural data flow are not covered by the arithmetic fix or this report.
The floor is the earliest stable release for this supported corpus, not a proof
about every earlier release, backend, shader or future regression.

## Reproduction

```sh
python3 test/graphics/check_slang_nonuniform_versions.py --family slang \
  --compiler /sdk/bin/slangc --output out/slang-version-results
```

[All 348 results](slang-minimum-version-results.csv) include both optimization
levels and the known failures. The script reuses the original corpus/parser and
adds arithmetic cases; raw SPIR-V, assembly, validation logs and commands remain
under `out/slang-minimum/<version>/results/`. Five boundary/default SDK archives came
from official shader-slang GitHub releases and were verified against release
SHA-256 metadata; 2026.18.3 reuses the previously verified official SDK.

## Dependency and configure-time policy

- CMake finds installed Slang >= 2026.18.1 and never downloads it.
- vcpkg uses the upstream `shader-slang` registry port instead of the project's
  2026.18.3 overlay. The registry snapshot keeps builds reproducible; the manifest
  requests the package by name. Upstream has no 2026.18.1 port entry, so CMake
  enforces the compiler floor rather than a manifest constraint on
  that unregistered version. Registry updates can select
  newer packages without changing the configure-time requirement.
- Version validation happens exclusively in CMake against the selected SDK's
  package version. Too-old or missing SDKs stop configuration with setup guidance.
  There is no C++ version parser or runtime compiler-version check.
- The lower-bound comparison allows newer calendar-year SDK versions, rather
  than treating the year as an incompatible major version. Deployment must use
  the selected SDK's libraries; no runtime mismatch detection is performed.

## Project validation at the minimum

The official 2026.18.1 macOS ARM64 SDK configured and built Sparkium CLI,
`test_slang_compiler` and `sparkium_fallback_test` in Ninja Release. Before the configure-only adjustment, compiler/API
coverage passed **4/4**, including the now-removed version parser test. Metal regression passed **27**
tests, with **12 conditional skips** for desktop/window opt-ins and unavailable
standalone ray-tracing pipeline parity; this was not a GUI test run. Three Blender scenes also completed native Metal
ray-query smoke runs at 96×96, 2 spp and 4 bounces with the minimum SDK; these
are compatibility checks, not repeat performance measurements.

CMake rejects the official 2026.18 SDK and accepts 2026.18.1 and the upstream
vcpkg 2026.18.2 SDK. The earlier runtime-version guard and its parser test have
been removed; compiler tests now contain the three shader/API tests only.

### Boundary SDK archive checksums

| Version | Official macOS ARM64 ZIP SHA-256 |
| --- | --- |
| 2026.10.2 | `5f37e80b16ee332669fa2355485f6cf2795fa5d406bf6e0b1533b3ea2f0e6d76` |
| 2026.11 | `595f06cb9c1c306609cd61d9ced20dc75d56c0153d7c1bf1e509834edf133c75` |
| 2026.18 | `83d4c3320d5ed87c10123e8253282bd68c3ab742561b1c71b8629a7679731c0c` |
| 2026.18.1 | `44fa1b6882c242a554b47cc11e1646758a229d53230ee90a62dd0e477dfac500` |
| 2026.18.2 | `cffb3a7eef12cabfc1e169271db4b7c9b982ed59aef6c3f138d02e392178bfbe` |

The default upstream registry SDK, 2026.18.2, was also run through the same
58-case compiler-output corpus with the same results as 2026.18.1 and 2026.18.3.

The upstream vcpkg registry installation of 2026.18.2 also configured and built
Sparkium CLI and the compiler tests successfully. After removing the runtime
version guard, the rebuilt compiler tests passed **3/3**,
and a Metal native-query Monster smoke run (96×96, 2 spp, 4 bounces) completed.
This checks the default package provider as well as the minimum external SDK.
