add_executable(test_rhi rhi_test.cpp)
target_link_libraries(test_rhi PRIVATE grassland_rhi GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_rhi)
