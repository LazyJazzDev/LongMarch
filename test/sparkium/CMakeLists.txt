add_executable(sparkium_fallback_test fallback_test.cpp metal_backend_test.cpp)
target_link_libraries(sparkium_fallback_test PRIVATE LongMarch GTest::gtest_main)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(sparkium_fallback_test)
add_test(NAME sparkium_fallback COMMAND sparkium_fallback_test)

# CPU/CUDA offline backends: Sobol rows, texture addressing, scene flattening and
# the CPU/CUDA image agreement. The scene-level comparison against the Vulkan
# rt_fallback reference is scripts/check_offline_backends.py.
add_executable(sparkium_offline_backend_test offline_backend_test.cpp)
target_link_libraries(sparkium_offline_backend_test PRIVATE LongMarch GTest::gtest_main)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(sparkium_offline_backend_test)
add_test(NAME sparkium_offline_backend COMMAND sparkium_offline_backend_test)
