file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")
add_library(sparkium_raytracing ${SOURCES})
target_link_libraries(sparkium_raytracing PUBLIC Grassland Snowberg sparkium_core)
