# VS Code C++ navigation

Open the repository root in VS Code. The checked-in `.vscode` configuration uses
**clangd** for C/C++ definitions, references, completion, and background indexing,
and **CMake Tools** for build integration. Both extensions are recommended in
`.vscode/extensions.json`. On this Mac they are installed, and `clangd` uses the
Apple Command Line Tools executable on `PATH` (`/usr/bin/clangd`).

The language server reads `cmake-build-metal/compile_commands.json`, generated
from the actual Metal/Vulkan build. This preserves target-specific include
paths, generated headers, C++17, arm64 options, and backend macros. A recursive
include-path list cannot reliably replace these compilation settings.

For an already configured build, refresh the database using the VS Code task
**LongMarch: refresh Metal compile commands**, or run:

```sh
cmake -S . -B cmake-build-metal -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

On a new checkout, first configure the dependencies and backend as described in
[metal-backend.md](metal-backend.md), using `cmake-build-metal` as the build
directory. Build once so that generated headers exist. CMake Tools is configured
to export the database on future reconfiguration; opening the workspace does
not automatically reconfigure the project. **Cmd+Shift+B** builds the Sparkium
CLI, GUI, and tests after refreshing the database.

Use **Go to Definition** (F12), **Go to References** (Shift+F12), or their context
menu actions. Initial background indexing takes time. After first installing
extensions, run **Developer: Reload Window** if the open editor has not activated
clangd. After switching branches or build options, refresh the database and use
**clangd: Restart language server** if necessary. The Output panel's **clangd**
channel shows compiler commands and index progress.

The current database covers files compiled by the active build, including Metal
and Vulkan. D3D12 and disabled CUDA/Python source paths need a build database for
their corresponding platform/options to obtain complete semantic navigation.
HLSL is not indexed as C++ by clangd.

If another machine uses a different build directory, change both
`cmake.buildDirectory` and clangd's `--compile-commands-dir` in workspace settings.
If Microsoft's C/C++ extension is also enabled, disable its IntelliSense for
this workspace to avoid competing language services; it is not required here.
Generated compilation databases and clangd caches are intentionally untracked.

Validated with Apple clangd 21 against 275 compilation commands: the language
server resolved `raytracing::Render` from `sparkium/core/core.cpp` to its
implementation in `pipelines/raytracing/core/core.cpp` and returned five
references across two source files. These two files and the native Metal AS
implementation produced no error diagnostics. Local protocol results are saved
in `out/vscode-validation/navigation-results.json`.
