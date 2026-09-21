add_library(sparkium_scene scene_definition.cpp)
target_link_libraries(sparkium_scene PUBLIC grassland_math)
if (LONGMARCH_CUDA_ENABLED)
    # CUDA device linking retains math_mesh_sdf's differential-kernel references.
    target_link_libraries(sparkium_scene PRIVATE grassland_physics_diff_kernel)
endif ()
if (LONGMARCH_PYTHON_ENABLED)
    # grassland math headers expose pybind11 when this project option is enabled.
    target_link_libraries(sparkium_scene PUBLIC Python3::Python)
endif ()
list(APPEND SPARKIUM_LIBS sparkium_scene)
set(SPARKIUM_LIBS ${SPARKIUM_LIBS} PARENT_SCOPE)
