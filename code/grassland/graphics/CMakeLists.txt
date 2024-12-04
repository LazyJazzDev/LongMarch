file(GLOB_RECURSE SOURCES "*.cpp" "*.h")

add_library(${GRASSLAND_SUBLIB_NAME} ${SOURCES})

list(APPEND GRASSLAND_LIBS ${GRASSLAND_SUBLIB_NAME})
set(GRASSLAND_LIBS ${GRASSLAND_LIBS} PARENT_SCOPE)

target_include_directories(${GRASSLAND_SUBLIB_NAME} PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(${GRASSLAND_SUBLIB_NAME} PUBLIC grassland_vulkan)

if (WIN32)
    target_link_libraries(${GRASSLAND_SUBLIB_NAME} PUBLIC grassland_d3d12)
endif ()
