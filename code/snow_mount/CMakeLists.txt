file(GLOB SOURCES "*.cpp")

add_library(SnowMount ${SOURCES})

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        set(SNOW_MOUNT_SUBLIB_NAME snow_mount_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

message(STATUS "SNOW_MOUNT_LIBS: ${SNOW_MOUNT_LIBS}")

target_link_libraries(SnowMount PUBLIC ${SNOW_MOUNT_LIBS})
