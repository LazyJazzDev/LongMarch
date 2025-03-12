list(APPEND LONGMARCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})
list(APPEND LONGMARCH_INCLUDE_DIR ${INC_DIR})
set(LONGMARCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)

add_library(LongMarch long_march.cpp)

add_subdirectory(grassland)
add_subdirectory(snow_mount)

target_include_directories(LongMarch PUBLIC ${LONGMARCH_INCLUDE_DIR})
target_link_libraries(LongMarch PUBLIC SnowMount)

pybind11_add_module(long_march long_march_pybind.cpp)
target_include_directories(long_march PUBLIC ${LONGMARCH_INCLUDE_DIR})
target_link_libraries(long_march PUBLIC LongMarch)

install( TARGETS long_march
    LIBRARY DESTINATION ${Python3_SITEARCH})
