file(GLOB SOURCES "*.cpp" "*.h")

target_sources(grassland_graphics PRIVATE ${SOURCES})

if (LONGMARCH_VULKAN_ENABLED)
    add_subdirectory(vulkan)
endif ()

if (LONGMARCH_D3D12_ENABLED)
    add_subdirectory(d3d12)
endif ()
