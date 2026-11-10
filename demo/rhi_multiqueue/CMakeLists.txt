add_executable(${DEMO_NAME} main.cpp)
target_link_libraries(${DEMO_NAME} PRIVATE grassland_rhi)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
