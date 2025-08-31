file(GLOB SOURCES "*.cpp")

if (LONGMARCH_CUDA_ENABLED)
    foreach (source_file ${SOURCES})
        set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
    endforeach ()
endif ()

add_library(grassland_physics_diff_kernel ${SOURCES})

target_include_directories(grassland_physics_diff_kernel PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(grassland_physics_diff_kernel PUBLIC grassland_util grassland_math)
