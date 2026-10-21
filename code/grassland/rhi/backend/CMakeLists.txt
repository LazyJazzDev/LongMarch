file(GLOB SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")

target_sources(grassland_rhi PRIVATE ${SOURCES})

if (LONGMARCH_VULKAN_ENABLED)
    add_subdirectory(vulkan)
endif ()

if (LONGMARCH_D3D12_ENABLED)
    add_subdirectory(d3d12)
endif ()

if (LONGMARCH_METAL_ENABLED)
    add_subdirectory(metal)
endif ()
