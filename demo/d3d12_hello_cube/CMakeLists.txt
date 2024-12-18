if (WIN32)
    file(GLOB_RECURSE DEMO_SOURCES "*.cpp" "*.h")

    add_executable(${DEMO_NAME} ${DEMO_SOURCES})

    target_link_libraries(${DEMO_NAME} LongMarch)

    PACK_SHADER_CODE(${DEMO_NAME})
endif ()
