file(GLOB SOURCES "*.cpp" "*.h")

list(APPEND LONGMARCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})
list(APPEND LONGMARCH_INCLUDE_DIR ${INC_DIR})
set(LONGMARCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)

add_library(LongMarch ${SOURCES})

add_subdirectory(grassland)

add_subdirectory(snow_mount)

target_include_directories(LongMarch PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(LongMarch PUBLIC Grassland SnowMount)
