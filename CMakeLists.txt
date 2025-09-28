cmake_minimum_required(VERSION 3.25)

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

if (UNIX AND NOT APPLE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif ()

if (CUDAToolkit_FOUND)
    # If CUDA Version >= 13.0
    set(CUDA_INC_DIR ${CUDAToolkit_INCLUDE_DIR})
    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
        message(STATUS "CUDA Version >= 13.0 detected. Thrust has moved to cccl subdirectory.")
        foreach (CUDAToolkit_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS})
            message(STATUS "CUDAToolkit_INCLUDE_DIR: ${CUDAToolkit_INCLUDE_DIR}")

            set(CUDA_INC_DIR ${CUDAToolkit_INCLUDE_DIR}/cccl)
            list(APPEND INC_LIST ${CUDA_INC_DIR})
        endforeach ()
    endif ()
    set(CUDART_LIB_NAME CUDA::cudart_static)
    list(APPEND LIB_LIST ${CUDART_LIB_NAME})
    set(LONGMARCH_CUDA_RUNTIME ON)
    if (NOT LONGMARCH_DISABLE_CUDA)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(LONGMARCH_CUDA_ENABLED ON)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        set(CMAKE_CUDA_ARCHITECTURES 75)
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
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

find_package(Vulkan COMPONENTS dxc)
if (Vulkan_FOUND)
    set(LONGMARCH_VULKAN_ENABLED ON)
    set(VULKAN_LIB_NAME ${Vulkan_LIBRARIES})
    list(APPEND LIB_LIST ${VULKAN_LIB_NAME})

    find_package(VulkanMemoryAllocator CONFIG REQUIRED)
    set(VMA_LIB_NAME Vulkan::Headers GPUOpen::VulkanMemoryAllocator)
    list(APPEND LIB_LIST ${VMA_LIB_NAME})
endif ()


find_package(glm CONFIG REQUIRED)
set(GLM_LIB_NAME glm::glm)
list(APPEND LIB_LIST ${GLM_LIB_NAME})

add_library(imgui)
target_sources(imgui
    PRIVATE
        external/imgui/imgui.cpp
        external/imgui/imgui_draw.cpp
        external/imgui/imgui_widgets.cpp
        external/imgui/imgui_demo.cpp
        external/imgui/imgui_tables.cpp)
target_include_directories(imgui
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/external/imgui>
        $<INSTALL_INTERFACE:include>)
set(IMGUI_LIB_NAME imgui)
list(APPEND LIB_LIST ${IMGUI_LIB_NAME})

find_package(mikktspace CONFIG REQUIRED)
set(MIKKTSPACE_LIB_NAME mikktspace::mikktspace)
list(APPEND LIB_LIST ${MIKKTSPACE_LIB_NAME})

find_package(tinyobjloader CONFIG REQUIRED)
set(TINYOBJLOADER_LIB_NAME tinyobjloader::tinyobjloader)
list(APPEND LIB_LIST ${TINYOBJLOADER_LIB_NAME})

find_package(Stb REQUIRED)
set(STB_INC_DIR ${Stb_INCLUDE_DIR})
list(APPEND INC_LIST ${STB_INC_DIR})

find_package(assimp CONFIG REQUIRED)
set(ASSIMP_LIB_NAME assimp::assimp)
list(APPEND LIB_LIST ${ASSIMP_LIB_NAME})

if (WIN32)
    set(LONGMARCH_D3D12_ENABLED ON)
    find_path(D3DX12_INCLUDE_DIRS "d3dx12.h")
    list(APPEND INC_LIST ${D3DX12_INCLUDE_DIRS})
endif ()

if (LONGMARCH_VULKAN_ENABLED)
    set(ENABLE_SPIRV_CODEGEN ON)
    set(DXC_LIB_NAME Vulkan::dxc_lib)
    list(APPEND LIB_LIST ${DXC_LIB_NAME})

    get_filename_component(VULKAN_DXC_DIR "${Vulkan_dxc_EXECUTABLE}" DIRECTORY)

    set(DXC_DLL_DIR "${VULKAN_DXC_DIR}")
elseif (LONGMARCH_D3D12_ENABLED)
    set(DXC_LIB_NAME dxcompiler)
    list(APPEND LIB_LIST ${DXC_LIB_NAME})

    if(DEFINED ENV{WindowsSdkVerBinPath})
        # e.g., C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\
        set(_BASE "$ENV{WindowsSdkVerBinPath}")
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(WINSDK_BIN_DIR "${_BASE}x64")
        else()
            set(WINSDK_BIN_DIR "${_BASE}x86")
        endif()
    endif()

    set(DXC_DLL_DIR "${WINSDK_BIN_DIR}")
endif ()

message(STATUS "DXC_LIB_NAME: ${DXC_LIB_NAME}")
message(STATUS "DXC_DLL_DIR: ${DXC_DLL_DIR}")

set(LONGMARCH_IMGUI_DIR ${CMAKE_CURRENT_LIST_DIR}/external/imgui)

message(STATUS "PROJECT_BINARY_DIR : ${PROJECT_BINARY_DIR}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
message(STATUS "Python3_RUNTIME_LIBRARY_DIRS: ${Python3_RUNTIME_LIBRARY_DIRS}")
message(STATUS "Python3_VERSION: ${Python3_VERSION}")
message(STATUS "Python3_SITEARCH: ${Python3_SITEARCH}")

add_subdirectory(external/pybind11)
set(PYBIND11_LIB_NAME pybind11::headers)
list(APPEND LIB_LIST ${PYBIND11_LIB_NAME})

add_subdirectory(scripts)
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
        if(CMAKE_GENERATOR MATCHES "^Visual Studio")
            message(STATUS "Using a Visual Studio generator")
            add_subdirectory(demo)
            add_subdirectory(test)
        else ()
            add_subdirectory(demo EXCLUDE_FROM_ALL)
            add_subdirectory(test EXCLUDE_FROM_ALL)
        endif()
    endif ()
endif ()


get_target_property(bin_path LongMarch INTERFACE_LINK_LIBRARIES)
message(STATUS "bin_path : ${bin_path}")
