file(GLOB SOURCES "*.cpp" "*.h" "*.mm")

target_sources(grassland_graphics PRIVATE ${SOURCES})
target_include_directories(grassland_graphics PUBLIC ${LONGMARCH_METAL_CPP_DIR})
target_compile_definitions(grassland_graphics PUBLIC LONGMARCH_METAL_ENABLED)
target_link_libraries(grassland_graphics PUBLIC
        "-framework Metal" "-framework Foundation" "-framework QuartzCore" "-framework Cocoa"
        spirv-cross-msl spirv-cross-glsl spirv-cross-core)
