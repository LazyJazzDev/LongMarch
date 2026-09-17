file(GLOB SOURCES "*.cpp" "*.h")

if (LONGMARCH_CUDA_ENABLED)
    # The CUDA backend is an optional part of this library: without the CUDA
    # toolkit the CPU backend and the shared shading core are still built.
    file(GLOB CUDA_SOURCES "*.cu" "*.cuh")
endif ()

add_library(${SPARKIUM_SUBLIB_NAME} ${SOURCES} ${CUDA_SOURCES})

target_link_libraries(${SPARKIUM_SUBLIB_NAME} PUBLIC
        Grassland Snowberg sparkium_core sparkium_entity sparkium_geometry sparkium_material)

if (LONGMARCH_CUDA_ENABLED)
    target_link_libraries(${SPARKIUM_SUBLIB_NAME} PUBLIC ${CUDART_LIB_NAME})
endif ()

list(APPEND SPARKIUM_LIBS ${SPARKIUM_SUBLIB_NAME})
set(SPARKIUM_LIBS ${SPARKIUM_LIBS} PARENT_SCOPE)
