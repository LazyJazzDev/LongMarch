file(GLOB SOURCES "*.cpp" "*.h")

add_library(${SPARKIUM_RASTER_SUBLIB_NAME} ${SOURCES})

target_link_libraries(${SPARKIUM_RASTER_SUBLIB_NAME} PUBLIC grassland_rhi grassland_math grassland_util sparkium_core)
list(APPEND SPARKIUM_RASTER_LIBS ${SPARKIUM_RASTER_SUBLIB_NAME})
set(SPARKIUM_RASTER_LIBS ${SPARKIUM_RASTER_LIBS} PARENT_SCOPE)
