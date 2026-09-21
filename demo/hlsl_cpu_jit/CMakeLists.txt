if (NOT TARGET slang::slang OR NOT TARGET slang::slang-llvm)
    message(STATUS "Skipping ${DEMO_NAME}: Slang with embedded LLVM is unavailable")
    return()
endif ()

add_executable(${DEMO_NAME} main.cpp)
target_link_libraries(${DEMO_NAME} PRIVATE slang::slang)
target_compile_definitions(${DEMO_NAME} PRIVATE
        DEMO_HLSL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernel.hlsl")

# The LLVM plugin is loaded dynamically, so link dependency discovery misses it.
add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:slang::slang>" "$<TARGET_FILE:slang::slang-llvm>"
        "$<TARGET_FILE_DIR:${DEMO_NAME}>")
