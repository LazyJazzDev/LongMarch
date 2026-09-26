# Apple metal-cpp headers are supplied by vcpkg or an external Apple SDK.
option(LONGMARCH_ENABLE_METAL "Build the native metal-cpp backend on macOS" ${APPLE})
if (LONGMARCH_ENABLE_METAL AND APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(LONGMARCH_METAL_ENABLED ON)
    find_path(LONGMARCH_METAL_CPP_DIR Metal/Metal.hpp
            PATH_SUFFIXES metal-cpp
            DOC "Directory containing Apple metal-cpp headers")
    if (NOT LONGMARCH_METAL_CPP_DIR)
        message(FATAL_ERROR "Install the vcpkg metal feature or set LONGMARCH_METAL_CPP_DIR to Apple metal-cpp headers. CMake will not download them.")
    endif ()
    get_filename_component(_metal_sdk_lib "${Vulkan_LIBRARY}" DIRECTORY)
    find_path(SPIRV_CROSS_INCLUDE_DIR spirv_cross/spirv_msl.hpp
            HINTS ${Vulkan_INCLUDE_DIRS} "$ENV{VULKAN_SDK}/include" REQUIRED)
    foreach (_part msl glsl core)
        find_library(SPIRV_CROSS_${_part}_LIBRARY NAMES spirv-cross-${_part}
                HINTS "${_metal_sdk_lib}" "$ENV{VULKAN_SDK}/lib" REQUIRED)
    endforeach ()
endif ()
