add_executable(${DEMO_NAME} main.cpp ../rhix_hello/save_image.cpp)
target_link_libraries(${DEMO_NAME} PRIVATE grassland_rhi grassland_util)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
