add_library(sparkium_raytracing raytracing.cpp)

# cpu first: it holds the compiled shaders, depends on nothing else here, and
# core links it to run the CPU execution path. The sublibrary name is normally
# assigned by the loop below, so it is set here too.
set(SPARKIUM_RAYTRACING_SUBLIB_NAME sparkium_raytracing_cpu)
add_subdirectory(cpu)

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    # check whether ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt exists
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
            AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt
            AND NOT subdir STREQUAL "cpu")
        set(SPARKIUM_RAYTRACING_SUBLIB_NAME sparkium_raytracing_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

target_link_libraries(sparkium_raytracing PUBLIC ${SPARKIUM_RAYTRACING_LIBS} Grassland Snowberg)
