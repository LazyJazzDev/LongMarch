# Dependency overlays

The manifest uses these ports rather than CMake-managed downloads.

`shader-slang` now comes from the upstream git registry selected in
`vcpkg-configuration.json`, with a minimum of 2026.18.1. The default registry
baseline currently selects 2026.18.2. There is no Slang overlay or exact-version
requirement; CMake accepts newer external SDKs and runtime initialization checks
the actual loaded compiler version. See the NonUniform boundary-version report
in `docs/reports/slang-minimum-version.md` before changing the minimum.

- `metal-cpp`: Apple's macOS 15 / iOS 18 header-only archive, unchanged from the
  version previously fetched by CMake, installed under `include/metal-cpp`.

Default manifest features `slang` and `metal` select the registry SDK and header overlay. External
SDK users can disable default features and enable only the ones they need.
The Slang SDK retains its own licenses installed by the port, independently of
its packaging recipe's MIT license.
