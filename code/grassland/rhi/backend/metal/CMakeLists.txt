file(GLOB SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h" "*.mm")
target_sources(grassland_rhi PRIVATE ${SOURCES})
target_include_directories(grassland_rhi PRIVATE ${LONGMARCH_METAL_CPP_DIR})
target_compile_definitions(grassland_rhi PUBLIC LONGMARCH_METAL_ENABLED)
target_link_libraries(grassland_rhi PRIVATE
        "-framework Metal" "-framework Foundation" "-framework QuartzCore" "-framework Cocoa")
