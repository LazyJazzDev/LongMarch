add_executable(sparkium_fallback_test fallback_test.cpp code_lines_test.cpp scene_window_test.cpp)
target_link_libraries(sparkium_fallback_test PRIVATE Sparkium grassland_rhi GTest::gtest_main)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(sparkium_fallback_test)
add_test(NAME sparkium_fallback COMMAND sparkium_fallback_test)

add_executable(sparkium_rhi_test rhi_binding_test.cpp)
target_link_libraries(sparkium_rhi_test PRIVATE Sparkium grassland_rhi GTest::gtest_main)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(sparkium_rhi_test)
add_test(NAME sparkium_rhi COMMAND sparkium_rhi_test)

add_executable(sparkium_browser_test browser_renderer_test.cpp ../../demo/sparkium_gui/browser_renderer.cpp)
target_link_libraries(sparkium_browser_test PRIVATE Sparkium grassland_rhi GTest::gtest_main)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(sparkium_browser_test)
add_test(NAME sparkium_browser COMMAND sparkium_browser_test)

add_executable(sparkium_worker_test render_worker_test.cpp ../../demo/thin_lens/thin_lens_scene.cpp
    ../../demo/practium_hello/practium_scene.cpp)
target_link_libraries(sparkium_worker_test PRIVATE Practium Sparkium grassland_rhi GTest::gtest_main)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(sparkium_worker_test)
add_test(NAME sparkium_worker COMMAND sparkium_worker_test)
