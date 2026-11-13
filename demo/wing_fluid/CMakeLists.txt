add_executable(${DEMO_NAME} main.cpp simulation_compute.cpp)
target_link_libraries(${DEMO_NAME} PRIVATE grassland_rhi grassland_util)
if (LONGMARCH_CUDA_ENABLED)
    target_sources(${DEMO_NAME} PRIVATE simulation.cu)
    target_link_libraries(${DEMO_NAME} PRIVATE CUDA::cudart_static)
endif ()
foreach (SOURCE display.slang wing_profile.h simulation.slang)
    set(HEADER ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.h)
    add_custom_command(OUTPUT ${HEADER}
        COMMAND "$<TARGET_FILE:simple_xxd>" ${SOURCE} ${HEADER}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS simple_xxd ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
    target_sources(${DEMO_NAME} PRIVATE ${HEADER})
endforeach ()
target_include_directories(${DEMO_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
