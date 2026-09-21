file(GLOB SOURCES "*.cpp" "*.h")

add_library(${SPARKIUM_CPU_TRACING_SUBLIB_NAME} ${SOURCES})

target_link_libraries(${SPARKIUM_CPU_TRACING_SUBLIB_NAME} PUBLIC sparkium_backend Grassland Snowberg sparkium_cpu_tracing_core sparkium_geometry)
list(APPEND SPARKIUM_CPU_TRACING_LIBS ${SPARKIUM_CPU_TRACING_SUBLIB_NAME})
set(SPARKIUM_CPU_TRACING_LIBS ${SPARKIUM_CPU_TRACING_LIBS} PARENT_SCOPE)
