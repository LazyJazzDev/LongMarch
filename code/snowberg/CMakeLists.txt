file(GLOB SOURCES "*.cpp")

add_library(Snowberg ${SOURCES})

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        set(SNOWBERG_SUBLIB_NAME snowberg_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

message(STATUS "SNOWBERG_LIBS: ${SNOWBERG_LIBS}")

target_link_libraries(Snowberg PUBLIC ${SNOWBERG_LIBS})
