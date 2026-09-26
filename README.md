# Long March

<!-- TOC -->
* [Long March](#long-march)
  * [Description](#description)
  * [How to Build](#how-to-build)
    * [Asset downloads (Git LFS)](#asset-downloads-git-lfs)
    * [Windows](#windows)
    * [Linux](#linux)
    * [macOS](#macos)
<!-- TOC -->

## Description

**LongMarch (长征)** is an integration of rendering and physics simulation algorithms.

This library provides solution as the underlying engine for games or other interactive applications (e.g. robotic
simulation).

**Components:**

- **Grassland (草地):** Basic libraries for math, physics, and wrapping graphics APIs (Vulkan, D3D12 & Metal).
- **Snowberg (雪山):** [NOT AVAILABLE: Still under planning] Planned to be a functional layer for supporting higher
  applications.
- **Sparkium   (星火):** The renderer library, a multi-pipeline renderer with common front-end interface. (Now supports
  rasterization and path tracing)
- **Contradium (矛盾):** [Work in progress] The physics simulation library.
- **Practium   (实践):**: An simulation engine for robotics and other applications.

Sparkium also has an experimental [compute ray tracing fallback](docs/rt-fallback.md)
for GPUs without hardware ray tracing. It shares the path tracer's materials,
lighting and JSON scenes and builds its acceleration structures with compute shaders.

Shaders are compiled with Slang 2026.18.1 or newer, checked at CMake configure time.
The default vcpkg feature supplies the upstream registry SDK;
CMake can also use an external SDK and never downloads Slang itself; see [shader compilation and migration](docs/slang-shaders.md).

## How to Build

We strongly recommend using [CLion](https://www.jetbrains.com/clion/) as the IDE for development. It has great CMake support for editing, building, and debugging.

[Visual Studio](https://visualstudio.microsoft.com/), [VSCode](https://code.visualstudio.com/), or other IDEs are also fine.

### Asset downloads (Git LFS)

The `assets` submodule uses [Git LFS](https://git-lfs.com/) for models, textures,
fonts, and binary data. Install Git LFS before cloning with submodules:

- Windows: install Git LFS using the installer linked above.
- macOS: `brew install git-lfs`.
- Ubuntu/Debian: `sudo apt install git-lfs`.

Then run `git lfs install` once for your user account. The clone and submodule
commands below will download the assets required by the checked-out version.

For an existing checkout, run from the repository root:

```bash
git lfs install
git submodule sync -- assets
git submodule update --init --recursive
git -C assets lfs pull
```

To defer large asset downloads, use `GIT_LFS_SKIP_SMUDGE=1` with the clone or
submodule update command (Bash/zsh), then run `git -C assets lfs pull` when needed.
On PowerShell, set `$env:GIT_LFS_SKIP_SMUDGE = "1"` before cloning and remove it
with `Remove-Item Env:GIT_LFS_SKIP_SMUDGE` afterward. Build and run demos only
after downloading their assets; LFS pointer files are not usable models or images.

Use `git submodule update` to follow the asset version pinned by this repository.
`git submodule update --remote` instead follows an upstream branch and can select
a different asset version.

The asset URL now points to `LazyJazzDev/LongMarchAssetsLFS`, which starts with
fresh history. The original `LongMarchAssets` repository is retained for older
LongMarch commits. Run `git submodule sync -- assets` after switching between
those revisions, before updating the submodule. Existing clones may retain old
Git objects locally; fresh clones do not download the old asset repository.

### Windows

#### Step 0: Prerequisites

- [vcpkg](https://github.com/microsoft/vcpkg): The C++ package manager. Clone the vcpkg repo to anywhere you like, we will refer tha vcpkg path as
  `<VCPKG_ROOT>` in the following instructions (the path ends in `vcpkg`, not its parent directory).
- [MSVC with Windows SDK (version 10+)](https://visualstudio.microsoft.com/downloads/): We usually install this via Visual Studio installer. You should select the following workloads during installation:
  - Desktop development with C++

  Then everything should be installed automatically.
- [[optional] Python3](https://python.org): We provide python package with pybind11. Such functionality requires Python3 installation. You may install anywhere you like (System-wide, User-only, Conda, Homebrew, etc.). We will refer the python executable path as `<PYTHON_EXECUTABLE_PATH>` in the following instructions.
- [[optional] Vulkan SDK](https://vulkan.lunarg.com/sdk/home): Vulkan is the latest cross-platform graphics API. Since D3D12 is available on Windows, this is optional. Install the SDK [Caution: not the Runtime (RT)] via the official **SDK installer**. You should be able to run `vulkaninfo` command in a new terminal after installation. **No optional components are needed for this project**.
- [[optional] CUDA Toolkit](https://developer.nvidia.com/cuda-downloads): CUDA is optional, however, some functions such as most of the GPU-accelerated physics simulation features will require CUDA. Install the toolkit with the official **exe (local)** installer. You should be able to run `nvcc --version` command in a new terminal after installation.

#### Step 1: Clone the repo

- Clone this repo with submodules:
  ```bash
  git clone --recurse-submodules
  ```
  or
- Clone without submodules:
  ```bash
  git clone <this-repo-url>
  ```
  Then initialize and update the submodules (in the root directory of this repo):
  ```bash
  git submodule update --init --recursive
  ```

#### Step 2: CMake Configuration

In the cloned repo root directory, apply cmake configuration with the following command:

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DPYTHON3_EXECUTABLE=<PYTHON_EXECUTABLE_PATH>
```

In this process, the CMake script will check whether you have installed Vulkan SDK and CUDA Toolkit, and configure the build options accordingly.

#### Step 3: Build your first demo

Run the following command to build the `hello triangle` demo:

```bash
cmake --build build --target demo_graphics_hello
```

Run with `--module triangle`, or omit `--module` to select one of eleven demos in the terminal menu.
The code is under [demo/graphics_hello/modules/triangle](demo/graphics_hello/modules/triangle);
see [launcher usage](demo/graphics_hello/README.md) for all modules and options.

The compiled executable should be located at `build/demo/graphics_hello/<Debug|Release>/demo_graphics_hello.exe`.

### Linux

For optional native Wayland support, Vulkan HDR10 output, and X11 compatibility,
see [Linux HDR presentation](docs/linux-hdr.md).

#### Step 0: Prerequisites

- [vcpkg](https://github.com/microsoft/vcpkg): The C++ package manager. Clone the vcpkg repo to anywhere you like, we will refer tha vcpkg path as
  `<VCPKG_ROOT>` in the following instructions (the path ends in `vcpkg`, not its parent directory).
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home): Vulkan is the latest cross-platform graphics API. Install the lastest Vulkan SDK via Tarball file, follow [this guide](https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html). You should be able to run `vulkaninfo` command in a new terminal after installation.
- [[optional] Python3](https://python.org): We provide python package with pybind11. Such functionality requires Python3 installation. You may install anywhere you like (System-wide, User-only, Conda, Homebrew, etc.). We will refer the python executable path as `<PYTHON_EXECUTABLE_PATH>` in the following instructions.
- [[optional] CUDA Toolkit](https://developer.nvidia.com/cuda-downloads): CUDA is optional, however, some functions such as most of the GPU-accelerated physics simulation features will require CUDA. Install the toolkit following the official instructions. You should be able to run `nvcc --version` command in a new terminal after installation.

#### Step 1: Clone the repo

- Clone this repo with submodules:
  ```bash
  git clone --recurse-submodules
  ```
  or
- Clone without submodules:
  ```bash
  git clone <this-repo-url>
  ```
  Then initialize and update the submodules (in the root directory of this repo):
  ```bash
  git submodule update --init --recursive
  ```

#### Step 2: CMake Configuration

In the cloned repo root directory, apply cmake configuration with the following command:

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DPYTHON3_EXECUTABLE=<PYTHON_EXECUTABLE_PATH>
```

In this process, the CMake script will check whether you have installed Vulkan SDK and CUDA Toolkit, and configure the build options accordingly.

#### Step 3: Build your first demo

Run the following command to build the `hello triangle` demo:

```bash
cmake --build build --target demo_graphics_hello
```

Run with `--module triangle`, or omit `--module` to select one of eleven demos in the terminal menu.
The code is under [demo/graphics_hello/modules/triangle](demo/graphics_hello/modules/triangle);
see [launcher usage](demo/graphics_hello/README.md) for all modules and options.

The compiled executable should be located at `build/demo/graphics_hello/demo_graphics_hello`.


### macOS

Apple Silicon builds now default to the native **metal-cpp** backend for Sparkium.
See [Metal backend setup and validation](docs/metal-backend.md) for build options,
API selection, and the compute ray tracing path.

#### Step 0: Prerequisites

- [vcpkg](https://github.com/microsoft/vcpkg): The C++ package manager. Clone the vcpkg repo to anywhere you like, we will refer tha vcpkg path as
  `<VCPKG_ROOT>` in the following instructions (the path ends in `vcpkg`, not its parent directory).
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home): Vulkan is the latest cross-platform graphics API. Install the SDK [Caution: not the Runtime (RT)] via the official **SDK installer**. You should be able to run `vulkaninfo` command in a new terminal after installation. **No optional components are needed for this project**.
- [[optional] Python3](https://python.org): We provide python package with pybind11. Such functionality requires Python3 installation. You may install anywhere you like (System-wide, User-only, Conda, Homebrew, etc.). We will refer the python executable path as `<PYTHON_EXECUTABLE_PATH>` in the following instructions.

CUDA is not available on macOS since Apple has deprecated NVIDIA GPU support.

#### Step 1: Clone the repo

- Clone this repo with submodules:
  ```bash
  git clone --recurse-submodules
  ```
  or
- Clone without submodules:
  ```bash
  git clone <this-repo-url>
  ```
  Then initialize and update the submodules (in the root directory of this repo):
  ```bash
  git submodule update --init --recursive
  ```

#### Step 2: CMake Configuration

In the cloned repo root directory, apply cmake configuration with the following command:

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DPYTHON3_EXECUTABLE=<PYTHON_EXECUTABLE_PATH>
```

In this process, the CMake script will check whether you have installed Vulkan SDK and CUDA Toolkit, and configure the build options accordingly.

#### Step 3: Build your first demo

Run the following command to build the `hello triangle` demo:

```bash
cmake --build build --target demo_graphics_hello
```

Run with `--module triangle`, or omit `--module` to select one of eleven demos in the terminal menu.
The code is under [demo/graphics_hello/modules/triangle](demo/graphics_hello/modules/triangle);
see [launcher usage](demo/graphics_hello/README.md) for all modules and options.

The compiled executable should be located at `build/demo/graphics_hello/demo_graphics_hello`.
