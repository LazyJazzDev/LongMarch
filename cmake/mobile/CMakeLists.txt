cmake_minimum_required(VERSION 3.25)
project(LongMarchMobile LANGUAGES C CXX)

# Library-only entry point. Native application projects live outside LongMarch.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(LONGMARCH_OFFLINE_SHADERS "Load prepared shaders without Slang or SPIRV-Cross" ON)
set(LONGMARCH_HEADLESS ON)
get_filename_component(LONGMARCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(LONGMARCH_INCLUDE_DIR "${LONGMARCH_ROOT}/code")
set(LONGMARCH_ASSETS_DIR "")

# Header-only dependencies may come from a host installation. Never link its binaries
# into a cross-compiled target. Callers can supply these directories explicitly.
set(LONGMARCH_MOBILE_HEADERS "" CACHE PATH "Portable headers: fmt, glm and eigen3")
if(NOT LONGMARCH_MOBILE_HEADERS)
    find_path(LONGMARCH_MOBILE_HEADERS_DISCOVERED fmt/format.h
        HINTS "${LONGMARCH_ROOT}/out/mobile/deps/arm64-osx/include"
        NO_CMAKE_FIND_ROOT_PATH REQUIRED)
    set(LONGMARCH_MOBILE_HEADERS "${LONGMARCH_MOBILE_HEADERS_DISCOVERED}")
endif()
foreach(header fmt/format.h glm/glm.hpp eigen3/Eigen/Core)
    if(NOT EXISTS "${LONGMARCH_MOBILE_HEADERS}/${header}")
        message(FATAL_ERROR "Missing ${header} in LONGMARCH_MOBILE_HEADERS")
    endif()
endforeach()
add_library(longmarch_mobile_headers INTERFACE)
target_include_directories(longmarch_mobile_headers SYSTEM INTERFACE
    "${LONGMARCH_MOBILE_HEADERS}" "${LONGMARCH_MOBILE_HEADERS}/eigen3")
target_compile_definitions(longmarch_mobile_headers INTERFACE FMT_HEADER_ONLY)
set(FMT_LIB_NAME longmarch_mobile_headers)

set(LONGMARCH_MOBILE_BACKEND "Auto" CACHE STRING "Mobile RHI backend: Auto, Metal or Vulkan")
set_property(CACHE LONGMARCH_MOBILE_BACKEND PROPERTY STRINGS Auto Metal Vulkan)
set(_mobile_backend "${LONGMARCH_MOBILE_BACKEND}")
if(_mobile_backend STREQUAL "Auto")
    if(APPLE)
        set(_mobile_backend Metal)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
        set(_mobile_backend Vulkan)
    endif()
endif()
if(_mobile_backend STREQUAL "Metal" AND APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(LONGMARCH_METAL_ENABLED ON)
    find_path(LONGMARCH_METAL_CPP_DIR Metal/Metal.hpp
        HINTS "${LONGMARCH_MOBILE_HEADERS}/metal-cpp"
        NO_CMAKE_FIND_ROOT_PATH REQUIRED)
elseif(_mobile_backend STREQUAL "Vulkan")
    set(LONGMARCH_VULKAN_ENABLED ON)
    if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
        find_library(VULKAN_LIB_NAME vulkan REQUIRED)
    elseif(NOT CMAKE_CROSSCOMPILING)
        # Host preparation/replay may use the same Vulkan backend as the device.
        find_package(Vulkan REQUIRED)
        set(VULKAN_LIB_NAME Vulkan::Vulkan)
    else()
        message(FATAL_ERROR "Vulkan mobile targets currently support OHOS only")
    endif()
    if(NOT EXISTS "${LONGMARCH_MOBILE_HEADERS}/vk_mem_alloc.h")
        message(FATAL_ERROR "Missing VulkanMemoryAllocator header")
    endif()
    add_library(longmarch_mobile_vma INTERFACE)
    target_include_directories(longmarch_mobile_vma INTERFACE "${LONGMARCH_MOBILE_HEADERS}")
    set(VMA_LIB_NAME longmarch_mobile_vma)
else()
    message(FATAL_ERROR "Select Metal on Apple, or Vulkan on OHOS/a native preparation host")
endif()

if(NOT LONGMARCH_OFFLINE_SHADERS)
    if(CMAKE_CROSSCOMPILING)
        message(FATAL_ERROR "Prepare shaders with a separate host build")
    endif()
    include("${LONGMARCH_ROOT}/cmake/Slang.cmake")
    foreach(part core glsl msl)
        find_package(spirv_cross_${part} CONFIG REQUIRED)
    endforeach()
endif()

option(LONGMARCH_MOBILE_IMGUI "Build the window-independent ImGui renderer" OFF)
if(LONGMARCH_MOBILE_IMGUI)
    set(LONGMARCH_IMGUI_SOURCE_DIR "${LONGMARCH_ROOT}/external/imgui" CACHE PATH "Dear ImGui sources")
    add_library(longmarch_mobile_imgui STATIC
        "${LONGMARCH_IMGUI_SOURCE_DIR}/imgui.cpp"
        "${LONGMARCH_IMGUI_SOURCE_DIR}/imgui_draw.cpp"
        "${LONGMARCH_IMGUI_SOURCE_DIR}/imgui_tables.cpp"
        "${LONGMARCH_IMGUI_SOURCE_DIR}/imgui_widgets.cpp"
        "${LONGMARCH_IMGUI_SOURCE_DIR}/imgui_demo.cpp")
    target_include_directories(longmarch_mobile_imgui PUBLIC "${LONGMARCH_IMGUI_SOURCE_DIR}")
endif()

set(GRASSLAND_SUBLIB_NAME grassland_util)
add_subdirectory("${LONGMARCH_ROOT}/code/grassland/util" util)
set(GRASSLAND_SUBLIB_NAME grassland_rhi)
add_subdirectory("${LONGMARCH_ROOT}/code/grassland/rhi" rhi)
target_link_libraries(grassland_rhi PUBLIC longmarch_mobile_headers)
add_library(LongMarch::Mobile ALIAS grassland_rhi)

option(LONGMARCH_MOBILE_TESTS "Build host shader target tests" OFF)
if(LONGMARCH_MOBILE_TESTS AND NOT LONGMARCH_OFFLINE_SHADERS)
    add_executable(longmarch_shader_target_test tests/shader_target.cpp)
    target_link_libraries(longmarch_shader_target_test PRIVATE LongMarch::Mobile)
endif()

if(LONGMARCH_MOBILE_TESTS AND LONGMARCH_MOBILE_IMGUI AND NOT LONGMARCH_OFFLINE_SHADERS)
    add_executable(longmarch_imgui_test tests/imgui.cpp)
    target_link_libraries(longmarch_imgui_test PRIVATE LongMarch::Mobile)
endif()
