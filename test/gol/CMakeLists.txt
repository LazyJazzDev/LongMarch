# Keep these CPU-only tests in C++; LONGMARCH_ADD_TEST marks .cpp files as CUDA.
file(GLOB GOL_TEST_SOURCES "*.cpp")

target_sources(test_main PRIVATE ${GOL_TEST_SOURCES} ../../demo/gol/game_of_life_lib/game_of_life_lib.cpp)
target_compile_definitions(test_main PRIVATE GOL_PATTERN_FILE="${CMAKE_SOURCE_DIR}/demo/gol/patterns/295P5H1V1.cells")
