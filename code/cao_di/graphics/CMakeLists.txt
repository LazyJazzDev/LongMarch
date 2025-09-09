file(GLOB SOURCES "*.cpp" "*.h")

add_library(${CAO_DI_SUBLIB_NAME} ${SOURCES})
add_subdirectory(backend)

list(APPEND CAO_DI_LIBS ${CAO_DI_SUBLIB_NAME})
set(CAO_DI_LIBS ${CAO_DI_LIBS} PARENT_SCOPE)

target_include_directories(${CAO_DI_SUBLIB_NAME} PUBLIC ${CHANGZHENG_INCLUDE_DIR} ${STB_INC_DIR})

target_link_libraries(${CAO_DI_SUBLIB_NAME} PUBLIC cao_di_imgui cao_di_vulkan ${DXC_LIB_NAME})

if (WIN32)
    target_link_libraries(${CAO_DI_SUBLIB_NAME} PUBLIC cao_di_d3d12)
endif ()

PACK_SHADER_CODE(${CAO_DI_SUBLIB_NAME})
