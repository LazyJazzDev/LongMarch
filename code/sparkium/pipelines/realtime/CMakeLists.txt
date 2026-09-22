file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")
add_library(sparkium_realtime ${SOURCES})
target_link_libraries(sparkium_realtime PUBLIC Grassland Snowberg sparkium_core)
