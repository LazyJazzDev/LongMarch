add_executable(test_tui text_test.cpp util_test.cpp)
target_link_libraries(test_tui PRIVATE snowberg_tui GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_tui)

add_executable(tui_menu_probe menu_probe.cpp)
target_link_libraries(tui_menu_probe PRIVATE snowberg_tui)

add_executable(tui_core_probe core_probe.cpp)
target_link_libraries(tui_core_probe PRIVATE snowberg_tui)
