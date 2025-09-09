if (WIN32)
    file(GLOB_RECURSE SOURCES "*.cpp" "*.h")

    add_library(${CAO_DI_SUBLIB_NAME} ${SOURCES})

    list(APPEND CAO_DI_LIBS ${CAO_DI_SUBLIB_NAME})
    set(CAO_DI_LIBS ${CAO_DI_LIBS} PARENT_SCOPE)

    target_include_directories(${CAO_DI_SUBLIB_NAME} PUBLIC ${CHANGZHENG_INCLUDE_DIR} ${Vulkan_INCLUDE_DIRS})

    target_link_libraries(${CAO_DI_SUBLIB_NAME} PUBLIC dxgi d3d12 dxguid d3dcompiler dxcompiler ${GLFW3_LIB_NAME} ${FMT_LIB_NAME} ${GLSLANG_LIB_NAME} ${GLM_LIB_NAME} ${IMGUI_LIB_NAME} cao_di_util cao_di_imgui)
endif ()
