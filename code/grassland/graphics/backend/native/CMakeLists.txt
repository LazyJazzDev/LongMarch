find_path(LONGMARCH_OPTIX_INCLUDE_DIR optix.h
        HINTS "${OptiX_ROOT}" "$ENV{OptiX_ROOT}" "$ENV{OPTIX_ROOT}"
        PATH_SUFFIXES include)
set(_optix_default OFF)
if (LONGMARCH_ENABLE_NATIVE_RENDER AND TARGET slang::slang AND TARGET slang::slang-llvm AND
        LONGMARCH_OPTIX_INCLUDE_DIR AND LONGMARCH_CUDA_ENABLED)
    set(_optix_default ON)
endif ()
option(LONGMARCH_ENABLE_OPTIX "Enable optional CUDA hardware ray tracing through OptiX" ${_optix_default})
if (LONGMARCH_ENABLE_OPTIX AND
        (NOT LONGMARCH_ENABLE_NATIVE_RENDER OR NOT TARGET slang::slang OR NOT TARGET slang::slang-llvm OR
         NOT LONGMARCH_CUDA_ENABLED OR NOT LONGMARCH_OPTIX_INCLUDE_DIR))
    message(FATAL_ERROR "OptiX requires native rendering, the Slang LLVM package, CUDA and SDK headers. Set OptiX_ROOT or LONGMARCH_OPTIX_INCLUDE_DIR, or disable LONGMARCH_ENABLE_OPTIX.")
endif ()
# Use one package for the compiler and LLVM plugin so their versions match.
if (LONGMARCH_ENABLE_NATIVE_RENDER AND TARGET slang::slang AND TARGET slang::slang-llvm)
    target_sources(grassland_graphics PRIVATE native_core.cpp native_shader.cpp native_shader_compat.cpp native_cpu_thread_pool.cpp)
    find_package(Threads REQUIRED)
    target_link_libraries(grassland_graphics PRIVATE slang::slang Threads::Threads)
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
elseif (LONGMARCH_ENABLE_NATIVE_RENDER)
    message(STATUS "Slang package with LLVM JIT not found: native CPU/CUDA rendering disabled")
endif ()
