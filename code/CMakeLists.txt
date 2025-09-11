list(APPEND LONGMARCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})
list(APPEND LONGMARCH_INCLUDE_DIR ${INC_DIR})
set(LONGMARCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)

add_library(LongMarch long_march.cpp)

add_subdirectory(grassland)
add_subdirectory(snowberg)
add_subdirectory(sparks)

target_include_directories(LongMarch PUBLIC ${LONGMARCH_INCLUDE_DIR})
target_link_libraries(LongMarch PUBLIC Grassland Snowberg Sparks)
