add_executable(${DEMO_NAME} main.cpp message_editor.cpp)
target_link_libraries(${DEMO_NAME} PRIVATE snowberg_tui)
