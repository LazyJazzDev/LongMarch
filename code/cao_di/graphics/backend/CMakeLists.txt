file(GLOB SOURCES "*.cpp" "*.h")

target_sources(cao_di_graphics PRIVATE ${SOURCES})

add_subdirectory(vulkan)

if (WIN32)
    add_subdirectory(d3d12)
endif ()
