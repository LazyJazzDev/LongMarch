file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")
add_library(sparkium_pipeline_common ${SOURCES})
target_link_libraries(sparkium_pipeline_common PUBLIC Grassland Snowberg sparkium_core)
