# Slang is optional: existing graphics backends remain buildable without it.
find_path(LONGMARCH_SLANG_INCLUDE_DIR slang.h HINTS "$ENV{VULKAN_SDK}/include/slang" /opt/vulkan/include/slang)
find_library(LONGMARCH_SLANG_LIBRARY NAMES slang-compiler HINTS "$ENV{VULKAN_SDK}/lib" /opt/vulkan/lib)
option(LONGMARCH_ENABLE_NATIVE_RENDER "Enable Slang CPU/CUDA compute backends" ON)
find_path(LONGMARCH_OPTIX_INCLUDE_DIR optix.h
        HINTS "${OptiX_ROOT}" "$ENV{OptiX_ROOT}" "$ENV{OPTIX_ROOT}"
        PATH_SUFFIXES include)
set(_optix_default OFF)
if (LONGMARCH_ENABLE_NATIVE_RENDER AND LONGMARCH_SLANG_INCLUDE_DIR AND LONGMARCH_SLANG_LIBRARY AND
        LONGMARCH_OPTIX_INCLUDE_DIR AND LONGMARCH_CUDA_ENABLED)
    set(_optix_default ON)
endif ()
option(LONGMARCH_ENABLE_OPTIX "Enable optional CUDA hardware ray tracing through OptiX" ${_optix_default})
if (LONGMARCH_ENABLE_OPTIX AND
        (NOT LONGMARCH_ENABLE_NATIVE_RENDER OR NOT LONGMARCH_SLANG_INCLUDE_DIR OR
         NOT LONGMARCH_SLANG_LIBRARY OR NOT LONGMARCH_CUDA_ENABLED OR NOT LONGMARCH_OPTIX_INCLUDE_DIR))
    message(FATAL_ERROR "OptiX requires native rendering, Slang, CUDA and OptiX SDK headers. Set OptiX_ROOT or LONGMARCH_OPTIX_INCLUDE_DIR, or disable LONGMARCH_ENABLE_OPTIX.")
endif ()
if (LONGMARCH_ENABLE_NATIVE_RENDER AND LONGMARCH_SLANG_INCLUDE_DIR AND LONGMARCH_SLANG_LIBRARY)
    target_sources(grassland_graphics PRIVATE native_core.cpp native_shader.cpp)
    target_include_directories(grassland_graphics PRIVATE ${LONGMARCH_SLANG_INCLUDE_DIR})
    target_link_libraries(grassland_graphics PRIVATE ${LONGMARCH_SLANG_LIBRARY})
    target_compile_definitions(grassland_graphics PUBLIC LONGMARCH_NATIVE_ENABLED)
    if (LONGMARCH_CUDA_ENABLED)
        target_compile_definitions(grassland_graphics PUBLIC LONGMARCH_NATIVE_CUDA_ENABLED)
        target_link_libraries(grassland_graphics PRIVATE CUDA::cuda_driver CUDA::nvrtc)
        if (UNIX AND NOT APPLE AND TARGET CUDA::nvrtc_builtins)
            # NVRTC dlopens its builtins by SONAME. An executable's RUNPATH is
            # not inherited by that dlopen, so retain the companion library as
            # a direct dependency for non-system CUDA installations as well.
            target_link_libraries(grassland_graphics PRIVATE
                    "-Wl,--push-state,--no-as-needed" CUDA::nvrtc_builtins "-Wl,--pop-state")
        endif ()
    endif ()
    if (LONGMARCH_ENABLE_OPTIX)
        target_sources(grassland_graphics PRIVATE optix_backend.cpp)
        target_include_directories(grassland_graphics PRIVATE ${LONGMARCH_OPTIX_INCLUDE_DIR})
        target_compile_definitions(grassland_graphics PUBLIC LONGMARCH_OPTIX_ENABLED)
        list(GET CUDAToolkit_INCLUDE_DIRS 0 _native_cuda_include)
        target_compile_definitions(grassland_graphics PRIVATE
                LONGMARCH_OPTIX_INCLUDE_DIR="${LONGMARCH_OPTIX_INCLUDE_DIR}"
                LONGMARCH_NATIVE_CUDA_INCLUDE_DIR="${_native_cuda_include}")
        target_link_libraries(grassland_graphics PRIVATE ${CMAKE_DL_LIBS})
        message(STATUS "OptiX hardware tracing enabled: ${LONGMARCH_OPTIX_INCLUDE_DIR}")
    endif ()
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(grassland_graphics PRIVATE OpenMP::OpenMP_CXX)
    endif ()
elseif (LONGMARCH_ENABLE_NATIVE_RENDER)
    message(STATUS "Slang compiler not found: native CPU/CUDA rendering disabled")
endif ()
