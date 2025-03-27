cmake_minimum_required(VERSION 3.22)

if (VCPKG_PATH)
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_PATH}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif ()

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # message with error: You must set VCPKG_PATH to your vcpkg root directory
    message(FATAL_ERROR "You must set CMAKE_TOOLCHAIN_FILE to your vcpkg script.\nYou may use VCPKG_PATH pointing to your vcpkg root directory for convenience.\ne.g. cmake -S . -B build -DVCPKG_PATH=/path/to/vcpkg")
endif ()

project(LongMarch)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    set(CUDATOOLKIT_LIB_NAME CUDA::toolkit)
    list(APPEND LIB_LIST ${CUDATOOLKIT_LIB_NAME})
    set(LONGMARCH_CUDA_RUNTIME ON)
    if (NOT LONGMARCH_DISABLE_CUDA)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(LONGMARCH_CUDA_ENABLED ON)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        set(CMAKE_CUDA_ARCHITECTURES 61)
    endif ()
else ()
    message(STATUS "CUDA not found, skipping CUDA code.")
endif ()

if (MSVC)
    # /utf-8
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=\"/utf-8\">")
    # /bigobj
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=\"/bigobj\">")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4068")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=\"/wd4068\">")
endif ()


find_package(fmt CONFIG REQUIRED)
set(FMT_LIB_NAME fmt::fmt) # fmt::fmt is also available
list(APPEND LIB_LIST ${FMT_LIB_NAME})

find_package(Freetype REQUIRED)
set(FREETYPE_LIB_NAME Freetype::Freetype) # since CMake 3.10
list(APPEND LIB_LIST ${FREETYPE_LIB_NAME})

find_package(Eigen3 CONFIG REQUIRED)
set(EIGEN3_LIB_NAME Eigen3::Eigen)
list(APPEND LIB_LIST ${EIGEN3_LIB_NAME})

find_package(glfw3 CONFIG REQUIRED)
set(GLFW3_LIB_NAME glfw)
list(APPEND LIB_LIST ${GLFW3_LIB_NAME})

find_package(Vulkan REQUIRED)
set(VULKAN_LIB_NAME ${Vulkan_LIBRARIES})
list(APPEND LIB_LIST ${VULKAN_LIB_NAME})

find_package(VulkanMemoryAllocator CONFIG REQUIRED)
set(VMA_LIB_NAME Vulkan::Headers GPUOpen::VulkanMemoryAllocator)
list(APPEND LIB_LIST ${VMA_LIB_NAME})

find_package(glslang CONFIG REQUIRED)
set(GLSLANG_LIB_NAME glslang::OSDependent glslang::glslang glslang::MachineIndependent glslang::GenericCodeGen glslang::glslang-default-resource-limits glslang::SPVRemapper glslang::SPIRV) # glslang::OGLCompiler glslang::HLSL
list(APPEND LIB_LIST ${GLSLANG_LIB_NAME})

find_package(glm CONFIG REQUIRED)
set(GLM_LIB_NAME glm::glm)
list(APPEND LIB_LIST ${GLM_LIB_NAME})

find_package(imgui CONFIG REQUIRED)
set(IMGUI_LIB_NAME imgui::imgui)
list(APPEND LIB_LIST ${IMGUI_LIB_NAME})

find_package(mikktspace CONFIG REQUIRED)
set(MIKKTSPACE_LIB_NAME mikktspace::mikktspace)
list(APPEND LIB_LIST ${MIKKTSPACE_LIB_NAME})

find_package(tinyobjloader CONFIG REQUIRED)
set(TINYOBJLOADER_LIB_NAME tinyobjloader::tinyobjloader)
list(APPEND LIB_LIST ${TINYOBJLOADER_LIB_NAME})

# decide base on platform, whether it's macOS or not
if (APPLE)
    include(external/DirectXShaderCompiler/cmake/caches/PredefinedParams.cmake)
    add_subdirectory(external/DirectXShaderCompiler)
    set(DXC_LIB_NAME dxcompiler)
    list(APPEND LIB_LIST ${DXC_LIB_NAME})
else ()
    find_package(directx-dxc CONFIG REQUIRED)
    set(DXC_LIB_NAME Microsoft::DirectXShaderCompiler)
    list(APPEND LIB_LIST ${DXC_LIB_NAME})
endif ()

find_package(Stb REQUIRED)
set(STB_INC_DIR ${Stb_INCLUDE_DIR})
list(APPEND INC_LIST ${STB_INC_DIR})

if (WIN32)
    find_path(D3DX12_INCLUDE_DIRS "d3dx12.h")
    list(APPEND INC_LIST ${D3DX12_INCLUDE_DIRS})

    get_filename_component(DIRECTX_DXC_TOOL_PATH ${DIRECTX_DXC_TOOL} DIRECTORY)
    message(STATUS "DIRECTX_DXC_TOOL_PATH: ${DIRECTX_DXC_TOOL_PATH}")
endif ()

message(STATUS "CMAKE_ARCHIVE_OUTPUT_DIRECTORY : ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY : ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "PROJECT_BINARY_DIR : ${PROJECT_BINARY_DIR}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(PYTHON3_LIB_NAME Python3::Python)
list(APPEND LIB_LIST ${PYTHON3_LIB_NAME})


message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")

# find_package(pybind11 CONFIG)
add_subdirectory(external/pybind11)
set(PYBIND11_LIB_NAME pybind11::headers)
list(APPEND LIB_LIST ${PYBIND11_LIB_NAME})

include(scripts/pack_shaders.cmake)

set(LONGMARCH_ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets/)

add_subdirectory(code)

#get_target_property(SHARED_LIB_PATH LongMarch IMPORTED_LOCATION)
#message(STATUS "SHARED_LIB_PATH : ${SHARED_LIB_PATH}")
#message("Executable output folder: $<TARGET_FILE_DIR:LongMarch>")
#file(GET_RUNTIME_DEPENDENCIES)

# If current list is the root list
if ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    if (NOT LONGMARCH_NO_DEMO)
        add_subdirectory(demo)
        add_subdirectory(test)
    endif ()
endif ()


get_target_property(bin_path LongMarch INTERFACE_LINK_LIBRARIES)
message(STATUS "bin_path : ${bin_path}")
