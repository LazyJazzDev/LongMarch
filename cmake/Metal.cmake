# Apple metal-cpp is header-only. Pin the official SDK archive for reproducible builds.
option(LONGMARCH_ENABLE_METAL "Build the native metal-cpp backend on macOS" ${APPLE})
if (LONGMARCH_ENABLE_METAL AND APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(LONGMARCH_METAL_ENABLED ON)
    set(LONGMARCH_METAL_CPP_DIR "" CACHE PATH "Optional existing metal-cpp header directory")
    if (NOT LONGMARCH_METAL_CPP_DIR)
        include(FetchContent)
        FetchContent_Declare(metal_cpp
                URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip
                URL_HASH SHA256=0433df1e0ab13c2b0becbd78665071e3fa28381e9714a3fce28a497892b8a184
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(metal_cpp)
        set(LONGMARCH_METAL_CPP_DIR "${metal_cpp_SOURCE_DIR}")
    endif ()
    get_filename_component(_metal_sdk_lib "${Vulkan_LIBRARY}" DIRECTORY)
    find_path(SPIRV_CROSS_INCLUDE_DIR spirv_cross/spirv_msl.hpp
            HINTS ${Vulkan_INCLUDE_DIRS} "$ENV{VULKAN_SDK}/include" REQUIRED)
    foreach (_part msl glsl core)
        find_library(SPIRV_CROSS_${_part}_LIBRARY NAMES spirv-cross-${_part}
                HINTS "${_metal_sdk_lib}" "$ENV{VULKAN_SDK}/lib" REQUIRED)
    endforeach ()
endif ()
