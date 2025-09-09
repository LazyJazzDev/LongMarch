file(GLOB SOURCES "*.cpp")

#if (LONGMARCH_CUDA_ENABLED)
#    message(STATUS "CUDA enabled, marking source files as CUDA source.")
#    foreach (source_file ${SOURCES})
#        set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
#    endforeach ()
#endif ()

add_subdirectory(diff_kernel)

add_library(${CAO_DI_SUBLIB_NAME} ${SOURCES})

list(APPEND CAO_DI_LIBS ${CAO_DI_SUBLIB_NAME})
set(CAO_DI_LIBS ${CAO_DI_LIBS} PARENT_SCOPE)

target_include_directories(${CAO_DI_SUBLIB_NAME} PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(${CAO_DI_SUBLIB_NAME} PUBLIC cao_di_util cao_di_math cao_di_physics_diff_kernel)
