file(GLOB SOURCES "*.cpp" "*.h")

add_library(${SPARKIUM_SUBLIB_NAME} ${SOURCES})

target_link_libraries(${SPARKIUM_SUBLIB_NAME} PUBLIC grassland_rhi grassland_math grassland_util sparkium_raster sparkium_raytracing)
list(APPEND SPARKIUM_LIBS ${SPARKIUM_SUBLIB_NAME})
set(SPARKIUM_LIBS ${SPARKIUM_LIBS} PARENT_SCOPE)

target_compile_definitions(${SPARKIUM_SUBLIB_NAME} PUBLIC LONGMARCH_SPARKIUM_SHADERS="${LONGMARCH_SPARKIUM_SHADERS}")
