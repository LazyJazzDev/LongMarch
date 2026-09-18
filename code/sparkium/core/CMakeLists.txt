file(GLOB SOURCES "*.cpp" "*.h")

add_library(${SPARKIUM_SUBLIB_NAME} ${SOURCES})

# `sparkium_native` depends back on this target, which CMake resolves by
# repeating both static libraries on the link line.
target_link_libraries(${SPARKIUM_SUBLIB_NAME} PUBLIC Grassland Snowberg sparkium_raster sparkium_raytracing
        sparkium_native)
list(APPEND SPARKIUM_LIBS ${SPARKIUM_SUBLIB_NAME})
set(SPARKIUM_LIBS ${SPARKIUM_LIBS} PARENT_SCOPE)

target_compile_definitions(${SPARKIUM_SUBLIB_NAME} PUBLIC LONGMARCH_SPARKIUM_SHADERS="${LONGMARCH_SPARKIUM_SHADERS}")
