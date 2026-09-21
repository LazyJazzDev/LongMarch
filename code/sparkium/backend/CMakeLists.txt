add_library(sparkium_backend device.cpp)
target_link_libraries(sparkium_backend PUBLIC grassland_graphics)
add_subdirectory(graphics)
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
    add_subdirectory(common)
    add_subdirectory(cpu)
    find_package(Threads REQUIRED)
    target_link_libraries(sparkium_backend PRIVATE slang::slang Threads::Threads)
    target_compile_definitions(sparkium_backend PUBLIC SPARKIUM_NATIVE_ENABLED)
    if (LONGMARCH_CUDA_ENABLED)
        add_subdirectory(cuda)
        target_compile_definitions(sparkium_backend PUBLIC SPARKIUM_NATIVE_CUDA_ENABLED)
        target_link_libraries(sparkium_backend PRIVATE CUDA::cuda_driver CUDA::nvrtc)
        if (UNIX AND NOT APPLE AND TARGET CUDA::nvrtc_builtins)
            # NVRTC dlopens its builtins by SONAME. An executable's RUNPATH is
            # not inherited by that dlopen, so retain the companion library as
            # a direct dependency for non-system CUDA installations as well.
            target_link_libraries(sparkium_backend PRIVATE
                    "-Wl,--push-state,--no-as-needed" CUDA::nvrtc_builtins "-Wl,--pop-state")
        endif ()
    endif ()
    if (LONGMARCH_ENABLE_OPTIX)

        target_include_directories(sparkium_backend PRIVATE ${LONGMARCH_OPTIX_INCLUDE_DIR})
        target_compile_definitions(sparkium_backend PUBLIC SPARKIUM_OPTIX_ENABLED)
        list(GET CUDAToolkit_INCLUDE_DIRS 0 _native_cuda_include)
        target_compile_definitions(sparkium_backend PRIVATE
                LONGMARCH_OPTIX_INCLUDE_DIR="${LONGMARCH_OPTIX_INCLUDE_DIR}"
                SPARKIUM_NATIVE_CUDA_INCLUDE_DIR="${_native_cuda_include}")
        target_link_libraries(sparkium_backend PRIVATE ${CMAKE_DL_LIBS})
        message(STATUS "OptiX hardware tracing enabled: ${LONGMARCH_OPTIX_INCLUDE_DIR}")
    endif ()
elseif (LONGMARCH_ENABLE_NATIVE_RENDER)
    message(STATUS "Slang package with LLVM JIT not found: native CPU/CUDA rendering disabled")
endif ()

list(APPEND SPARKIUM_LIBS sparkium_backend)
set(SPARKIUM_LIBS ${SPARKIUM_LIBS} PARENT_SCOPE)

# Algorithm libraries belong to backend implementations, not the public scene API.
add_subdirectory(graphics/raster)
add_subdirectory(graphics/path_tracing)
add_subdirectory(cpu/path_tracing)
add_subdirectory(cuda/path_tracing)
add_library(sparkium_cpu_bvh cpu/cpu_bvh.cpp)
target_link_libraries(sparkium_cpu_bvh PUBLIC grassland_math)
target_link_libraries(sparkium_cpu_tracing_core PUBLIC sparkium_cpu_bvh)

add_library(sparkium_shader_codegen common/shader_graph.cpp)
target_link_libraries(sparkium_shader_codegen PUBLIC sparkium_scene PRIVATE sparkium_core RapidJSON)

add_library(sparkium_graphics_renderer graphics/render_backend.cpp graphics/geometry.cpp graphics/texture.cpp graphics/scene_objects.cpp)
target_link_libraries(sparkium_graphics_renderer PUBLIC sparkium_scene
        PRIVATE sparkium_core sparkium_geometry sparkium_material sparkium_entity sparkium_shader_codegen sparkium_raytracing sparkium_raster)

add_library(sparkium_execution render_backend.cpp)
target_link_libraries(sparkium_execution PUBLIC sparkium_scene PRIVATE sparkium_backend sparkium_graphics_renderer)
if (LONGMARCH_ENABLE_NATIVE_RENDER AND TARGET slang::slang AND TARGET slang::slang-llvm)
    add_library(sparkium_cpu_renderer cpu/render_backend.cpp cpu/geometry.cpp cpu/texture.cpp cpu/scene_objects.cpp)
    target_link_libraries(sparkium_cpu_renderer PUBLIC sparkium_scene
            PRIVATE sparkium_core sparkium_geometry sparkium_material sparkium_entity sparkium_shader_codegen sparkium_cpu_tracing)
    target_link_libraries(sparkium_execution PRIVATE sparkium_cpu_renderer)
    if (LONGMARCH_CUDA_ENABLED)
        add_library(sparkium_cuda_renderer cuda/render_backend.cpp cuda/geometry.cpp cuda/texture.cpp cuda/scene_objects.cpp)
        target_link_libraries(sparkium_cuda_renderer PUBLIC sparkium_scene
                PRIVATE sparkium_core sparkium_geometry sparkium_material sparkium_entity sparkium_shader_codegen sparkium_cuda_tracing)
        target_link_libraries(sparkium_execution PRIVATE sparkium_cuda_renderer)
        if (LONGMARCH_ENABLE_OPTIX)
            target_include_directories(sparkium_cuda_renderer PRIVATE ${LONGMARCH_OPTIX_INCLUDE_DIR})
        endif ()
    endif ()
endif ()
