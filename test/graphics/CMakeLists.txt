# Native window tests require a logged-in macOS desktop; run explicitly.
if(APPLE)
    add_executable(test_window_gestures EXCLUDE_FROM_ALL window_gestures_test.mm)
    set_property(TARGET test_window_gestures PROPERTY OBJCXX_STANDARD 17)
    target_link_libraries(test_window_gestures PRIVATE LongMarch GTest::gtest_main "-framework Cocoa")
endif()

# Requires a desktop session; kept out of the default/headless test suite.
add_executable(test_window_input EXCLUDE_FROM_ALL window_input_test.cpp)
target_link_libraries(test_window_input PRIVATE LongMarch GTest::gtest_main)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(test_window_input)

if(LONGMARCH_VULKAN_ENABLED)
    add_executable(test_vulkan_compatibility EXCLUDE_FROM_ALL vulkan_compatibility_test.cpp)
    target_link_libraries(test_vulkan_compatibility PRIVATE Grassland GTest::gtest_main)
    LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(test_vulkan_compatibility)
endif()
