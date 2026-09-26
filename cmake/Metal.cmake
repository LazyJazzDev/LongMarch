# Apple metal-cpp headers are supplied by vcpkg or an external Apple SDK.
option(LONGMARCH_ENABLE_METAL "Build the native metal-cpp backend on macOS" ${APPLE})
if (LONGMARCH_ENABLE_METAL AND APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(LONGMARCH_METAL_ENABLED ON)
    # Older builds cached an empty optional override. find_path treats that as
    # an existing result and skips discovery, even when vcpkg installed headers.
    if (DEFINED LONGMARCH_METAL_CPP_DIR AND "${LONGMARCH_METAL_CPP_DIR}" STREQUAL "")
        unset(LONGMARCH_METAL_CPP_DIR CACHE)
        unset(LONGMARCH_METAL_CPP_DIR)
    endif ()
    find_path(LONGMARCH_METAL_CPP_DIR Metal/Metal.hpp
            PATH_SUFFIXES metal-cpp
            DOC "Directory containing Apple metal-cpp headers")
    if (NOT LONGMARCH_METAL_CPP_DIR)
        message(FATAL_ERROR "Install the vcpkg metal feature or set LONGMARCH_METAL_CPP_DIR to Apple metal-cpp headers. CMake will not download them.")
    endif ()
    # Metal uses SPIRV-Cross as a shader translator, independently of Vulkan.
    # Resolve exported targets from vcpkg or an external CMAKE_PREFIX_PATH.
    foreach (_part core glsl msl)
        find_package(spirv_cross_${_part} CONFIG REQUIRED)
    endforeach ()
endif ()
