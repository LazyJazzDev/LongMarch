# Native window tests require a logged-in macOS desktop; run explicitly.
if(APPLE)
    add_executable(test_window_gestures EXCLUDE_FROM_ALL window_gestures_test.mm)
    set_property(TARGET test_window_gestures PROPERTY OBJCXX_STANDARD 17)
    target_link_libraries(test_window_gestures PRIVATE LongMarch GTest::gtest_main "-framework Cocoa")
endif()
