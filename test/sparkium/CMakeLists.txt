add_executable(sparkium_fallback_test fallback_test.cpp metal_backend_test.cpp cpu_backend_test.cpp)
target_link_libraries(sparkium_fallback_test PRIVATE LongMarch GTest::gtest_main)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(sparkium_fallback_test)
add_test(NAME sparkium_fallback COMMAND sparkium_fallback_test)

# The BVH oracle test includes the shader sources so it can use the shader's own
# traversal, which means it cannot also link the CPU backend's library: both
# would define them. It compiles the BVH builder directly instead.
add_executable(sparkium_cpu_bvh_test cpu_bvh_test.cpp
        ${CMAKE_SOURCE_DIR}/code/sparkium/pipelines/raytracing/cpu/bvh_builder.cpp)
target_include_directories(sparkium_cpu_bvh_test PRIVATE
        ${CMAKE_SOURCE_DIR}/code
        ${CMAKE_SOURCE_DIR}/code/sparkium/pipelines/raytracing/cpu/shaders
        ${CMAKE_BINARY_DIR}/code/sparkium/pipelines/raytracing/cpu/shaders)
target_link_libraries(sparkium_cpu_bvh_test PRIVATE Grassland GTest::gtest_main)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(sparkium_cpu_bvh_test)
add_test(NAME sparkium_cpu_bvh COMMAND sparkium_cpu_bvh_test)
