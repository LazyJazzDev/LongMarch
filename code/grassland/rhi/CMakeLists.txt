file(GLOB SOURCES "*.h" "*.cpp")

add_library(${GRASSLAND_SUBLIB_NAME} ${SOURCES})

list(APPEND GRASSLAND_LIBS ${GRASSLAND_SUBLIB_NAME})
set(GRASSLAND_LIBS ${GRASSLAND_LIBS} PARENT_SCOPE)

target_include_directories(${GRASSLAND_SUBLIB_NAME} PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(${GRASSLAND_SUBLIB_NAME} PRIVATE grassland_util ${GLFW3_LIB_NAME})

add_subdirectory(backend)

add_subdirectory(rhix)
