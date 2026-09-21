add_library(sparkium_cpu_tracing raytracing.cpp)

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    # check whether ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt exists
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
        set(SPARKIUM_CPU_TRACING_SUBLIB_NAME sparkium_cpu_tracing_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

target_link_libraries(sparkium_cpu_tracing PUBLIC ${SPARKIUM_CPU_TRACING_LIBS} Grassland Snowberg)
