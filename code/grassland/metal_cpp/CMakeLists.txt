if (LONGMARCH_METAL_ENABLED)
    # Both rendering APIs may coexist in one application. Emit metal-cpp's
    # implementation symbols once, independently of either renderer.
    add_library(grassland_metal_cpp STATIC metal_cpp.cpp)
    target_include_directories(grassland_metal_cpp PUBLIC ${LONGMARCH_METAL_CPP_DIR})
    target_link_libraries(grassland_metal_cpp PUBLIC
            "-framework Metal" "-framework Foundation" "-framework QuartzCore")
endif ()
