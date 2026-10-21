file(GLOB SOURCES CONFIGURE_DEPENDS "*.h" "*.cpp")
target_sources(grassland_rhi PRIVATE ${SOURCES})
