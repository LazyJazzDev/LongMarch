add_library(sparkium_renderer renderer.cpp)
target_link_libraries(sparkium_renderer PUBLIC sparkium_scene PRIVATE sparkium_execution sparkium_backend)
list(APPEND SPARKIUM_LIBS sparkium_renderer)
set(SPARKIUM_LIBS ${SPARKIUM_LIBS} PARENT_SCOPE)
