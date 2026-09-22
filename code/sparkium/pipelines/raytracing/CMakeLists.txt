add_library(sparkium_raytracing raytracing.cpp core/core.cpp core/scene.cpp core/film.cpp)
target_link_libraries(sparkium_raytracing PUBLIC sparkium_pipeline_common)
