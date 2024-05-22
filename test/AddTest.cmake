
function(ADD_TEST)
    file(GLOB_RECURSE SOURCES "*.h" "*.cpp")

    target_sources(test_main PRIVATE ${SOURCES})
endfunction()
