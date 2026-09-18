# Portable CPU/CUDA rendering backend. The shading kernels are transpiled
# from the HLSL sources shared with the GPU pipelines. Registered from
# pipelines/CMakeLists.txt, where ${SPARKIUM_LIBS} is consumed by the parent.

file(GLOB SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")

add_library(sparkium_portable ${SOURCES})

# The JIT-compiled kernels include hlsl_compat.h from the source tree.
target_compile_definitions(sparkium_portable PRIVATE
    LONGMARCH_PORTABLE_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/code")

target_link_libraries(sparkium_portable PUBLIC Grassland Snowberg sparkium_core sparkium_raytracing ${CMAKE_DL_LIBS})

if (LONGMARCH_CUDA_ENABLED)
    set_source_files_properties(kernel_cuda.cu PROPERTIES LANGUAGE CUDA)
    target_sources(sparkium_portable PRIVATE kernel_cuda.cu)
    target_compile_definitions(sparkium_portable PUBLIC SPARKIUM_PORTABLE_CUDA)
    target_link_libraries(sparkium_portable PUBLIC ${CUDART_LIB_NAME} cuda)
    target_include_directories(sparkium_portable PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif ()

list(APPEND SPARKIUM_LIBS sparkium_portable)
set(SPARKIUM_LIBS ${SPARKIUM_LIBS} PARENT_SCOPE)
