file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")

if (LONGMARCH_CUDA_ENABLED)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "*.cu")
    list(APPEND SOURCES ${CUDA_SOURCES})
endif ()

add_library(sparkium_native ${SOURCES})

target_link_libraries(sparkium_native PUBLIC Grassland Snowberg sparkium_core)

if (LONGMARCH_CUDA_ENABLED)
    # Selects the CUDA renderer in `native.cpp` and in the demos.
    target_compile_definitions(sparkium_native PUBLIC LONGMARCH_SPARKIUM_NATIVE_CUDA)
    set_target_properties(sparkium_native PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
