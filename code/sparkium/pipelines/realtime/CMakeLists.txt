file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")
add_library(sparkium_realtime ${SOURCES})
target_link_libraries(sparkium_realtime PUBLIC Grassland Snowberg sparkium_core)
target_compile_definitions(sparkium_realtime PRIVATE LONGMARCH_REALTIME_SHADERS="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
