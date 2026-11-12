set(DEMO_SOURCES main.cpp practium_scene.cpp practium_scene.h)

add_executable(${DEMO_NAME} ${DEMO_SOURCES})

target_link_libraries(${DEMO_NAME} PRIVATE Practium grassland_rhi)

LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
