file(GLOB SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")
if(LONGMARCH_HEADLESS)
    list(FILTER SOURCES EXCLUDE REGEX "/backend_window[^/]*\\.cpp$")
endif()

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

if (APPLE AND NOT LONGMARCH_HEADLESS)
    target_sources(grassland_rhi PRIVATE backend_window_cocoa.mm backend_window_gestures.mm)
    target_link_libraries(grassland_rhi PRIVATE "-framework Cocoa" "-framework QuartzCore")
endif ()
