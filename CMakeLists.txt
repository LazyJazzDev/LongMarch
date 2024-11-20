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
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(LONGMARCH_CUDA_ENABLED ON)
#    set_target_properties(my_cuda_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#    set_target_properties(my_cuda_app PROPERTIES CUDA_ARCHITECTURES "60;70;75")
else ()
    message(STATUS "CUDA not found, skipping CUDA code.")
endif ()

if (MSVC)
#    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
#    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/utf-8\"")
#    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler \"/source-charset:utf-8\",\"/execution-charset:utf-8\">")
endif()

find_package(fmt CONFIG REQUIRED)
set(FMT_LIB_NAME fmt::fmt) # fmt::fmt is also available
list(APPEND LIB_LIST ${FMT_LIB_NAME})

find_package(spdlog CONFIG REQUIRED)
set(SPDLOG_LIB_NAME spdlog::spdlog) # spdlog::spdlog is also available
list(APPEND LIB_LIST ${SPDLOG_LIB_NAME})

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

if (WIN32)
    find_path(D3DX12_INCLUDE_DIRS "d3dx12.h")
    list(APPEND INC_LIST ${D3DX12_INCLUDE_DIRS})
endif ()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

include(scripts/pack_shaders.cmake)

add_subdirectory(code)

# If current list is the root list
if ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    add_subdirectory(demo)
    add_subdirectory(test)
endif ()
