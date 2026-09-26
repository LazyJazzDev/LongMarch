# Shared mobile build dependencies

Install once on the host (Apple Silicon example):

```sh
/path/to/vcpkg/vcpkg install --x-manifest-root=platforms/mobile \
  --x-install-root=out/mobile/deps --triplet=arm64-osx
```

The iOS and HarmonyOS CMake projects automatically discover this location. Override
`LONGMARCH_MOBILE_DEPS` only when using a different installation directory. CMake
never downloads dependencies. This manifest uses the same Slang 2026.18.3 overlay
as the desktop build; the version is pinned to the mobile cache format. For an
external Slang SDK, install with `--x-no-default-features` and configure host
preparation with `-Dslang_DIR=/sdk/lib/cmake/slang`.

- Host preparation links Slang; Metal preparation also needs SPIRV-Cross from the
  external Vulkan SDK. Official Slang's macOS ARM64 library requires macOS 26+.
- Apps/replay builds define `LONGMARCH_OFFLINE_SHADERS`: neither Slang nor
  SPIRV-Cross is linked into them. They embed `.slang` source only to reproduce
  cache request keys, then load SPIR-V (HarmonyOS) or cached MSL (iOS).
- Portable headers and Apple metal-cpp headers come from vcpkg. FreeType 2.13.3
  and MikkTSpace sources come from the source-only overlay package and are
  compiled with the target SDK, so host libraries cannot enter the app.
- Existing DXC bundles are incompatible. Regenerate iOS resources, then use the
  HarmonyOS staging script with `harmony_fallback_prepare` to generate software
  ray-tracing variants as well. See each platform README for commands.

The old per-platform header manifests remain usable for explicit header overrides;
the shared manifest is the default setup for both apps.
