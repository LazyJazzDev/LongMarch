set(RHI_HELLO_MODULES window triangle cube texture blend nbody raytracing ray_query hdr imgui imgui_multi_window)
add_executable(${DEMO_NAME} main.cpp)
target_link_libraries(${DEMO_NAME} PRIVATE grassland_rhi grassland_util snowberg_tui)

foreach(MODULE IN LISTS RHI_HELLO_MODULES)
    set(MODULE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/modules/${MODULE})
    set(MODULE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/modules/${MODULE})
    target_sources(${DEMO_NAME} PRIVATE ${MODULE_SOURCE_DIR}/module.cpp)
    set_source_files_properties(${MODULE_SOURCE_DIR}/module.cpp PROPERTIES INCLUDE_DIRECTORIES ${MODULE_BINARY_DIR})
    # Each example embeds only its own shaders, with no shared demo runtime or shader registry.
    file(GLOB MODULE_SHADERS CONFIGURE_DEPENDS RELATIVE ${MODULE_SOURCE_DIR} ${MODULE_SOURCE_DIR}/shaders/*.slang)
    foreach(SHADER IN LISTS MODULE_SHADERS)
        set(HEADER ${MODULE_BINARY_DIR}/${SHADER}.h)
        file(MAKE_DIRECTORY ${MODULE_BINARY_DIR}/shaders)
        add_custom_command(OUTPUT ${HEADER}
            COMMAND "$<TARGET_FILE:simple_xxd>" ${SHADER} ${HEADER}
            WORKING_DIRECTORY ${MODULE_SOURCE_DIR}
            DEPENDS simple_xxd ${MODULE_SOURCE_DIR}/${SHADER}
        )
        target_sources(${DEMO_NAME} PRIVATE ${HEADER})
    endforeach()
endforeach()
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(${DEMO_NAME})
