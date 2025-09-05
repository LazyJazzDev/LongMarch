file(GLOB_RECURSE DEMO_SOURCES "*.cpp" "*.h")

add_executable(${DEMO_NAME} ${DEMO_SOURCES})

target_link_libraries(${DEMO_NAME} PUBLIC LongMarch)
