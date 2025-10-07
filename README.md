# Long March

<!-- TOC -->
* [Long March](#long-march)
  * [Description](#description)
  * [How to Build](#how-to-build)
    * [Windows](#windows)
    * [Linux](#linux)
    * [macOS](#macos)
<!-- TOC -->

## Description

**LongMarch (长征)** is an integration of rendering and physics simulation algorithms.

This library provides solution as the underlying engine for games or other interactive applications (e.g. robotic
simulation).

**Components:**

- **Grassland (草地):** Basic libraries for math, physics, and wrapping graphics APIs (Vulkan & D3D12).
- **Snowberg (雪山):** [NOT AVAILABLE: Still under planning] Planned to be a functional layer for supporting higher
  applications.
- **Sparkium   (星火):** The renderer library, a multi-pipeline renderer with common front-end interface. (Now supports
  rasterization and path tracing)
- **Contradium (矛盾):** [Work in progress] The physics simulation library.
- **Practium   (实践):**: An simulation engine for robotics and other applications.

## How to Build

We strongly recommend using [CLion](https://www.jetbrains.com/clion/) as the IDE for development. It has great CMake support for editing, building, and debugging.

[Visual Studio](https://visualstudio.microsoft.com/), [VSCode](https://code.visualstudio.com/), or other IDEs are also fine.

### Windows

#### Step 0: Prerequisites

- [Python3](https://python.org): Install anywhere you like (System-wide, User-only, Conda, Homebrew, etc.). We will refer the python executable path as `<PYTHON_EXECUTABLE_PATH>` in the following instructions.
- [vcpkg](https://github.com/microsoft/vcpkg): The C++ package manager. Clone the vcpkg repo to anywhere you like, we will refer tha vcpkg path as
  `<VCPKG_ROOT>` in the following instructions (the path ends in `vcpkg`, not its parent directory).
- [MSVC with Windows SDK (version 10+)](https://visualstudio.microsoft.com/downloads/): We usually install this via Visual Studio installer. You should select the following workloads during installation:
  - Desktop development with C++

  Then everything should be installed automatically.
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
cmake --build build --target demo_graphics_hello_triangle
```

the code is under path [demo/graphics_hello_triangle](demo/graphics_hello_triangle)

The compiled executable should be located at `build/demo/graphics_hello_triangle/<Debug|Release>/graphics_hello_triangle.exe`.

### Linux

#### Step 0: Prerequisites

- [Python3](https://python.org): Install anywhere you like (System-wide, User-only, Conda, Homebrew, etc.). We will refer the python executable path as `<PYTHON_EXECUTABLE_PATH>` in the following instructions.
- [vcpkg](https://github.com/microsoft/vcpkg): The C++ package manager. Clone the vcpkg repo to anywhere you like, we will refer tha vcpkg path as
  `<VCPKG_ROOT>` in the following instructions (the path ends in `vcpkg`, not its parent directory).
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home): Vulkan is the latest cross-platform graphics API. Install the lastest Vulkan SDK via Tarball file, follow [this guide](https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html). You should be able to run `vulkaninfo` command in a new terminal after installation.
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
cmake --build build --target demo_graphics_hello_triangle
```

the code is under path [demo/graphics_hello_triangle](demo/graphics_hello_triangle)

The compiled executable should be located at `build/demo/graphics_hello_triangle/graphics_hello_triangle`.


### macOS

#### Step 0: Prerequisites

- [Python3](https://python.org): Install anywhere you like (System-wide, User-only, Conda, Homebrew, etc.). We will refer the python executable path as `<PYTHON_EXECUTABLE_PATH>` in the following instructions.
- [vcpkg](https://github.com/microsoft/vcpkg): The C++ package manager. Clone the vcpkg repo to anywhere you like, we will refer tha vcpkg path as
  `<VCPKG_ROOT>` in the following instructions (the path ends in `vcpkg`, not its parent directory).
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home): Vulkan is the latest cross-platform graphics API. Install the SDK [Caution: not the Runtime (RT)] via the official **SDK installer**. You should be able to run `vulkaninfo` command in a new terminal after installation. **No optional components are needed for this project**.

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
cmake --build build --target demo_graphics_hello_triangle
```

the code is under path [demo/graphics_hello_triangle](demo/graphics_hello_triangle)

The compiled executable should be located at `build/demo/graphics_hello_triangle/graphics_hello_triangle`.
