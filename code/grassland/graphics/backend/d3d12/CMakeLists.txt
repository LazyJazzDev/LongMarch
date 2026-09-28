file(GLOB_RECURSE SOURCES "*.cpp" "*.h")

target_sources(grassland_graphics PRIVATE ${SOURCES})
target_link_libraries(grassland_graphics PUBLIC dxgi d3d12 dxguid ${GLFW3_LIB_NAME} ${FMT_LIB_NAME} ${GLM_LIB_NAME} ${IMGUI_LIB_NAME})
target_link_libraries(grassland_graphics PRIVATE ${EIGEN3_LIB_NAME})
target_compile_definitions(grassland_graphics PUBLIC -DLONGMARCH_D3D12_ENABLED)
