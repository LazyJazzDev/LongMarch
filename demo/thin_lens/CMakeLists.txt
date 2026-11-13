set(DEMO_SOURCES main.cpp thin_lens_scene.cpp thin_lens_scene.h)
add_executable(${DEMO_NAME} ${DEMO_SOURCES})
target_link_libraries(${DEMO_NAME} PRIVATE Sparkium grassland_rhi)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
