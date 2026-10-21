add_executable(${DEMO_NAME} main.cpp modules/window/module.cpp modules/triangle/module.cpp modules/cube/module.cpp modules/texture/module.cpp modules/blend/module.cpp modules/nbody/module.cpp)
target_link_libraries(${DEMO_NAME} PRIVATE grassland_rhi grassland_util)

PACK_SHADER_CODE(${DEMO_NAME})
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
