# Long March

## Description

This library is targeting to be an all-in-one library for graphics study purpose.

## Build

This project is built with CMake. It also requires Vulkan SDK and vcpkg.

To build the project, you need to do CMake configuration with setting VCPKG_PATH to the path of vcpkg.
```bash
cmake -B build -S . -DVCPKG_PATH=/path/to/vcpkg
```

Then, you can build the project with CMake.
```bash
cmake --build build
```
