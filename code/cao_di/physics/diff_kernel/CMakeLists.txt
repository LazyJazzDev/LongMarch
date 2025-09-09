file(GLOB SOURCES "*.cpp")

if (CHANGZHENG_CUDA_ENABLED)
    foreach (source_file ${SOURCES})
        set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
    endforeach ()
endif ()

add_library(cao_di_physics_diff_kernel ${SOURCES})

target_include_directories(cao_di_physics_diff_kernel PUBLIC ${CHANGZHENG_INCLUDE_DIR})

target_link_libraries(cao_di_physics_diff_kernel PUBLIC cao_di_util cao_di_math)
