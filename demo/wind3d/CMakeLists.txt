if (LONGMARCH_CUDA_ENABLED)
    add_executable(${DEMO_NAME} main.cpp simulation.cu geometry.cpp render.cpp)
    target_link_libraries(${DEMO_NAME} PRIVATE grassland_rhi grassland_util CUDA::cudart_static)
    set(HEADER ${CMAKE_CURRENT_BINARY_DIR}/display.slang.h)
    add_custom_command(OUTPUT ${HEADER}
            COMMAND "$<TARGET_FILE:simple_xxd>" display.slang ${HEADER}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS simple_xxd ${CMAKE_CURRENT_SOURCE_DIR}/display.slang)
    target_sources(${DEMO_NAME} PRIVATE ${HEADER})
    target_include_directories(${DEMO_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
endif ()
