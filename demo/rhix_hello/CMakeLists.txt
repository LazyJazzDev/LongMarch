add_executable(${DEMO_NAME} main.cpp save_image.cpp modules/window/module.cpp modules/triangle/module.cpp modules/cube/module.cpp modules/texture/module.cpp modules/blend/module.cpp modules/nbody/module.cpp modules/raytracing/module.cpp modules/ray_query/module.cpp modules/imgui/module.cpp modules/hdr/module.cpp)
target_link_libraries(${DEMO_NAME} PRIVATE grassland_rhi grassland_util snowberg_tui)

PACK_SHADER_CODE(${DEMO_NAME})
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
