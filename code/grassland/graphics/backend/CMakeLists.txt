file(GLOB SOURCES "*.cpp" "*.h")

target_sources(grassland_graphics PRIVATE ${SOURCES})

# The host backend has no optional dependency and is always available.
add_subdirectory(host)

if (LONGMARCH_VULKAN_ENABLED)
    add_subdirectory(vulkan)
endif ()

if (LONGMARCH_D3D12_ENABLED)
    add_subdirectory(d3d12)
endif ()

if (LONGMARCH_METAL_ENABLED)
    add_subdirectory(metal)
endif ()
