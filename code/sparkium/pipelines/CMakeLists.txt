add_subdirectory(raytracing)
add_subdirectory(realtime)

file(GLOB SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")

add_library(${SPARKIUM_SUBLIB_NAME} ${SOURCES})

target_link_libraries(${SPARKIUM_SUBLIB_NAME} PUBLIC Grassland Snowberg sparkium_raytracing sparkium_realtime)
list(APPEND SPARKIUM_LIBS ${SPARKIUM_SUBLIB_NAME})
set(SPARKIUM_LIBS ${SPARKIUM_LIBS} PARENT_SCOPE)
