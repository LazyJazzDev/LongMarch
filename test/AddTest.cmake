
function(ADD_TEST)
    file(GLOB_RECURSE SOURCES "*.cpp")

    if (CHANGZHENG_CUDA_ENABLED)
        foreach (source_file ${SOURCES})
            set_source_files_properties(${source_file} TARGET_DIRECTORY test_main PROPERTIES LANGUAGE CUDA)
            message(STATUS "CUDA enabled, marking source files as CUDA source. ${source_file}")
        endforeach ()

        file(GLOB_RECURSE CUDA_SOURCES "*.cu")

        target_sources(test_main PRIVATE ${CUDA_SOURCES})
    endif ()

    target_sources(test_main PRIVATE ${SOURCES})
endfunction()
