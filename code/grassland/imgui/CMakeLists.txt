include(${CMAKE_CURRENT_LIST_DIR}/float_vertices.cmake)
set(SOURCES "")
list(APPEND SOURCES ${LONGMARCH_IMGUI_DIR}/backends/imgui_impl_glfw.cpp)

if (LONGMARCH_VULKAN_ENABLED)
    make_float_vertex_backend(vulkan cpp BACKEND_SOURCE)
    list(APPEND SOURCES ${BACKEND_SOURCE})
endif ()

if (LONGMARCH_D3D12_ENABLED)
    make_float_vertex_backend(dx12 cpp BACKEND_SOURCE)
    list(APPEND SOURCES ${BACKEND_SOURCE})
endif ()

if (LONGMARCH_METAL_ENABLED)
    make_float_vertex_backend(metal mm BACKEND_SOURCE)
    list(APPEND SOURCES ${BACKEND_SOURCE})
endif ()
add_library(${GRASSLAND_SUBLIB_NAME} ${SOURCES})
if (LONGMARCH_METAL_ENABLED)
    target_include_directories(${GRASSLAND_SUBLIB_NAME} PRIVATE ${LONGMARCH_METAL_CPP_DIR})
    target_compile_definitions(${GRASSLAND_SUBLIB_NAME} PRIVATE IMGUI_IMPL_METAL_CPP)
    target_link_libraries(${GRASSLAND_SUBLIB_NAME} PUBLIC "-framework Metal" "-framework Foundation")
endif ()

list(APPEND GRASSLAND_LIBS ${GRASSLAND_SUBLIB_NAME})
set(GRASSLAND_LIBS ${GRASSLAND_LIBS} PARENT_SCOPE)

target_include_directories(${GRASSLAND_SUBLIB_NAME} PUBLIC ${LONGMARCH_INCLUDE_DIR} ${LONGMARCH_IMGUI_DIR}/backends)

if (LONGMARCH_VULKAN_ENABLED)
    target_include_directories(${GRASSLAND_SUBLIB_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS})
endif ()

target_link_libraries(${GRASSLAND_SUBLIB_NAME} PUBLIC ${IMGUI_LIB_NAME} ${VULKAN_LIB_NAME} ${GLFW3_LIB_NAME})
