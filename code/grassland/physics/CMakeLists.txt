
if (LONGMARCH_CUDA_ENABLED)
    message(STATUS "CUDA enabled, compiling CUDA code.")
    file(GLOB_RECURSE SOURCES "*.cu" "*.h")
else ()
    message(STATUS "CUDA disabled, compiling C++ code.")
    file(GLOB_RECURSE SOURCES "*.cpp" "*.h")
endif ()

add_library(${GRASSLAND_SUBLIB_NAME} ${SOURCES})

list(APPEND GRASSLAND_LIBS ${GRASSLAND_SUBLIB_NAME})
set(GRASSLAND_LIBS ${GRASSLAND_LIBS} PARENT_SCOPE)

target_include_directories(${GRASSLAND_SUBLIB_NAME} PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(${GRASSLAND_SUBLIB_NAME} PUBLIC grassland_util)
