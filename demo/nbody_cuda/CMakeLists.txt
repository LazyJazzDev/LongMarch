if (CHANGZHENG_CUDA_ENABLED)
    file(GLOB_RECURSE DEMO_SOURCES "*.cpp" "*.h" "*.cu" "*.cuh")

    add_executable(${DEMO_NAME} ${DEMO_SOURCES})

    target_link_libraries(${DEMO_NAME} ChangZheng)

    PACK_SHADER_CODE(${DEMO_NAME})
endif ()
