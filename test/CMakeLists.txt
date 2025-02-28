# Enumerate all subdirectories of current directory

find_package(GTest CONFIG REQUIRED)

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

include(AddTest.cmake)

add_executable(test_main test_main.cpp)

enable_testing()

foreach (subdir ${subdirs})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        set(TEST_NAME test_${subdir})
        add_subdirectory(${subdir})
    endif()
endforeach()

target_link_libraries(test_main PRIVATE LongMarch GTest::gtest GTest::gtest_main)


file(GLOB_RECURSE SOURCES "*.cpp")

foreach (source_file ${SOURCES})
    get_source_file_property(IS_CUDA_SOURCE ${source_file} LANGUAGE)
    message(STATUS "source_file : ${source_file} IS_CUDA_SOURCE : ${IS_CUDA_SOURCE}")
endforeach ()
