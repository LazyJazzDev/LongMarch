# Long March

## Description

This library is targeting to be an all-in-one library for graphics study purpose.

## Build

This project is built with CMake. It also requires Vulkan SDK and vcpkg.

To build the project, you need to do CMake configuration with setting VCPKG_PATH to the path of vcpkg.

```bash
cmake -B build -S . -DVCPKG_PATH=/path/to/vcpkg
```

**notice:** You may need to make sure that your vcpkg version is above 2025.3.19 for D3D12 compatability. Otherwise, the
integrated version of DirectX Shader Compiler(dxc) library cannot sign the compiled shader program without official
released `dxil.dll` placed under PATH directories, which you may find
at [official releases page](https://github.com/microsoft/DirectXShaderCompiler/releases) of dxc.

Then, you can build the project with CMake.

```bash
cmake --build build
```

### Specifying Python environment

To specify Python environment, you may need to set the following variables. Otherwise, CMake could refer to the python
environment installed by vcpkg.

```bash
-DPython3_EXECUTABLE=/path/to/python
```

With the above setting, the project could install compiled python module library files to the specified python
environment with CMake install target.
