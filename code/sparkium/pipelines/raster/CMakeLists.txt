add_library(sparkium_raster raster.cpp)

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    # check whether ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt exists
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
        set(SPARKIUM_RASTER_SUBLIB_NAME sparkium_raster_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

target_link_libraries(sparkium_raster PUBLIC ${SPARKIUM_RASTER_LIBS} Grassland Snowberg)
