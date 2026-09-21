add_library(sparkium_raytracing raytracing.cpp)

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    # check whether ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt exists
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
        set(SPARKIUM_RAYTRACING_SUBLIB_NAME sparkium_raytracing_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

target_link_libraries(sparkium_raytracing PUBLIC ${SPARKIUM_RAYTRACING_LIBS} Grassland Snowberg)
