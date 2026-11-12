file(GLOB_RECURSE SOURCES "*.cpp")

if (LONGMARCH_CUDA_ENABLED)
    foreach (source_file ${SOURCES})
        set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
    endforeach ()
endif ()

file(GLOB_RECURSE SOURCES "*.cpp" "*.cxx")

add_library(${GRASSLAND_SUBLIB_NAME} ${SOURCES})

list(APPEND GRASSLAND_LIBS ${GRASSLAND_SUBLIB_NAME})
set(GRASSLAND_LIBS ${GRASSLAND_LIBS} PARENT_SCOPE)

target_include_directories(${GRASSLAND_SUBLIB_NAME} PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(${GRASSLAND_SUBLIB_NAME} PUBLIC ${MIKKTSPACE_LIB_NAME} ${TINYOBJLOADER_LIB_NAME} grassland_util)
