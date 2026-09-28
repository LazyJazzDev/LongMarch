file(GLOB_RECURSE SOURCES "*.cpp" "*.h")

target_sources(grassland_graphics PRIVATE ${SOURCES})
target_include_directories(grassland_graphics PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(grassland_graphics PUBLIC ${VULKAN_LIB_NAME} ${GLFW3_LIB_NAME} ${FMT_LIB_NAME} ${VMA_LIB_NAME} ${GLM_LIB_NAME} ${IMGUI_LIB_NAME})
target_compile_definitions(grassland_graphics PUBLIC -DLONGMARCH_VULKAN_ENABLED)
