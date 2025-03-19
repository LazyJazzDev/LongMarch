file(GLOB_RECURSE SOURCES "*.cpp")

if (LONGMARCH_CUDA_ENABLED)
    # mark *.cpp as CUDA sources
    foreach (source IN LISTS SOURCES)
        if (source MATCHES ".*\\.cpp$")
            set_property(SOURCE ${source} PROPERTY LANGUAGE CUDA)
        endif ()
    endforeach ()
endif ()

add_library(${SNOW_MOUNT_SUBLIB_NAME} ${SOURCES})

list(APPEND SNOW_MOUNT_LIBS ${SNOW_MOUNT_SUBLIB_NAME})
set(SNOW_MOUNT_LIBS ${SNOW_MOUNT_LIBS} PARENT_SCOPE)

target_include_directories(${SNOW_MOUNT_SUBLIB_NAME} PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(${SNOW_MOUNT_SUBLIB_NAME} PUBLIC Grassland)

PACK_SHADER_CODE(${SNOW_MOUNT_SUBLIB_NAME})
