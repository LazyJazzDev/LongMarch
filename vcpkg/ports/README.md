# Dependency overlays

The manifest uses these ports rather than CMake-managed downloads.

- `shader-slang`: based on the upstream Microsoft/vcpkg `shader-slang`
  [2026.18.2 port](https://github.com/microsoft/vcpkg/tree/dc1232a6e05dcc49703091e83743e3b4df9b9b7c/ports/shader-slang) (MIT; see `shader-slang/LICENSE-port.txt`), updated to official
  2026.18.3 archives and SHA-512 checksums. Separate debug-symbol archives are
  omitted; release binaries are used for both configurations, as upstream does.
  Standard modules and signed macOS binaries retain upstream packaging behavior.
  All six archive hashes were also checked against the release's SHA-256 metadata.
  Supports Windows/Linux/macOS ARM64 and x86-64; this is not a mobile runtime SDK.
  Remove this overlay when the registry provides the tested version or a validated
  successor. Update the minimum in `cmake/Slang.cmake` together with compiler tests.
- `metal-cpp`: Apple's macOS 15 / iOS 18 header-only archive, unchanged from the
  version previously fetched by CMake, installed under `include/metal-cpp`.

Default manifest features `slang` and `metal` select these dependencies. External
SDK users can disable default features and enable only the ones they need.
The Slang SDK retains its own licenses installed by the port, independently of
its packaging recipe's MIT license.
