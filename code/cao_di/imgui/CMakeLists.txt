set(SOURCES "")
list(APPEND SOURCES ${CHANGZHENG_IMGUI_DIR}/backends/imgui_impl_glfw.cpp)
list(APPEND SOURCES ${CHANGZHENG_IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)

if (WIN32)
    list(APPEND SOURCES ${CHANGZHENG_IMGUI_DIR}/backends/imgui_impl_dx12.cpp)
endif ()

add_library(${CAO_DI_SUBLIB_NAME} ${SOURCES})

list(APPEND CAO_DI_LIBS ${CAO_DI_SUBLIB_NAME})
set(CAO_DI_LIBS ${CAO_DI_LIBS} PARENT_SCOPE)

target_include_directories(${CAO_DI_SUBLIB_NAME} PUBLIC ${CHANGZHENG_INCLUDE_DIR} ${Vulkan_INCLUDE_DIRS} ${CHANGZHENG_IMGUI_DIR}/backends)

target_link_libraries(${CAO_DI_SUBLIB_NAME} PUBLIC ${IMGUI_LIB_NAME} ${VULKAN_LIB_NAME} ${GLFW3_LIB_NAME})
