file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")
add_library(sparkium_raytracing ${SOURCES})
target_link_libraries(sparkium_raytracing PUBLIC Grassland Snowberg sparkium_core)
target_compile_definitions(sparkium_raytracing PRIVATE LONGMARCH_RAYTRACING_SHADERS="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
