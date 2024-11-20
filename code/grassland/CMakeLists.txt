file(GLOB SOURCES "*.cpp")

add_library(Grassland ${SOURCES})

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        set(GRASSLAND_SUBLIB_NAME grassland_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

message(STATUS "GRASSLAND_LIBS: ${GRASSLAND_LIBS}")

target_link_libraries(Grassland PUBLIC ${GRASSLAND_LIBS})
