
function(ADD_TEST)
    if (LONGMARCH_CUDA_ENABLED)
        file(GLOB_RECURSE SOURCES "*.h" "*.cpp" "*.cuh" "*.cu")
    else ()
        file(GLOB_RECURSE SOURCES "*.h" "*.cpp")
    endif ()

    target_sources(test_main PRIVATE ${SOURCES})
endfunction()
