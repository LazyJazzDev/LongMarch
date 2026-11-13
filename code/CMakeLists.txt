list(APPEND LONGMARCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})
list(APPEND LONGMARCH_INCLUDE_DIR ${INC_DIR})
set(LONGMARCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)

add_library(LongMarchCommon long_march.cpp)

add_subdirectory(grassland)
add_subdirectory(snowberg)
add_subdirectory(sparkium)
add_subdirectory(contradium)
add_subdirectory(practium)

target_include_directories(LongMarchCommon PUBLIC ${LONGMARCH_INCLUDE_DIR})
target_link_libraries(LongMarchCommon PUBLIC Grassland Snowberg Sparkium Contradium Practium)
add_library(LongMarch INTERFACE)
target_link_libraries(LongMarch INTERFACE LongMarchCommon)
