if (LONGMARCH_WIND3D_STANDALONE)
    add_executable(wind3d_task EXCLUDE_FROM_ALL task.cpp simulation.cu geometry.cpp)
    target_link_libraries(wind3d_task PRIVATE CUDA::cudart Threads::Threads)
    if (BUILD_TESTING)
        add_test(NAME wind3d_multigpu
            COMMAND ${CMAKE_COMMAND} -DWIND3D_TASK=$<TARGET_FILE:wind3d_task>
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_multigpu.cmake)
        set_tests_properties(wind3d_multigpu PROPERTIES TIMEOUT 1200 RUN_SERIAL TRUE)
    endif ()
    return()
endif ()

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
