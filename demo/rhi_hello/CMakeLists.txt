add_executable(${DEMO_NAME} main.cpp modules/hello_window/module.cpp modules/hello_triangle/module.cpp modules/hello_cube/module.cpp modules/hello_texture/module.cpp modules/hello_blend/module.cpp)
target_link_libraries(${DEMO_NAME} PRIVATE grassland_rhi grassland_util)

PACK_SHADER_CODE(${DEMO_NAME})
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
