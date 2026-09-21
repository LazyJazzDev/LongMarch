add_library(sparkium_renderer renderer.cpp scene_instance.cpp shader_graph.cpp)
target_link_libraries(sparkium_renderer PUBLIC sparkium_scene
        PRIVATE sparkium_core sparkium_geometry sparkium_material sparkium_entity RapidJSON)
list(APPEND SPARKIUM_LIBS sparkium_renderer)
set(SPARKIUM_LIBS ${SPARKIUM_LIBS} PARENT_SCOPE)
