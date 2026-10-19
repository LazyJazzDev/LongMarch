add_executable(test_rhi rhi_test.cpp object_ptr_test.cpp obj_ptr_test.cpp raw_api_test.cpp command_context_test.cpp window_test.cpp program_test.cpp buffer_test.cpp)
target_link_libraries(test_rhi PRIVATE grassland_rhi GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_rhi)

if (LONGMARCH_VULKAN_ENABLED)
    target_sources(test_rhi PRIVATE vulkan_enumeration_test.cpp vulkan_timeline_test.cpp)
    target_include_directories(test_rhi PRIVATE ${Vulkan_INCLUDE_DIRS})
endif ()

if (LONGMARCH_METAL_ENABLED)
    target_sources(test_rhi PRIVATE metal_test.cpp)
    target_include_directories(test_rhi PRIVATE ${LONGMARCH_METAL_CPP_DIR})
endif ()
