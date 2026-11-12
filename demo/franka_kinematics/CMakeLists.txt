file(GLOB_RECURSE DEMO_SOURCES "*.cpp" "*.h")

add_executable(${DEMO_NAME} ${DEMO_SOURCES})

target_link_libraries(${DEMO_NAME} PRIVATE Sparkium grassland_rhi ${ASSIMP_LIB_NAME})

LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
