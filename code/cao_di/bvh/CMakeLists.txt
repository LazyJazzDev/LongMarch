if (CHANGZHENG_CUDA_ENABLED)
    file(GLOB_RECURSE SOURCES "*.cpp" "*.cu")
else ()
    file(GLOB_RECURSE SOURCES "*.cpp")
endif ()

add_library(${CAO_DI_SUBLIB_NAME} ${SOURCES})

list(APPEND CAO_DI_LIBS ${CAO_DI_SUBLIB_NAME})
set(CAO_DI_LIBS ${CAO_DI_LIBS} PARENT_SCOPE)

target_include_directories(${CAO_DI_SUBLIB_NAME} PUBLIC ${CHANGZHENG_INCLUDE_DIR})

target_link_libraries(${CAO_DI_SUBLIB_NAME} PUBLIC ${EIGEN3_LIB_NAME} cao_di_util)
