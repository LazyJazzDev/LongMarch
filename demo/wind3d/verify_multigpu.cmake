# Run on an allocated eight-GPU node: ctest --test-dir cmake-build-release -R wind3d
foreach (mode full final)
    set(export_options)
    if (mode STREQUAL final)
        list(APPEND export_options --no-export)
    endif ()
    foreach (devices 1 2 4 8)
        set(snapshot "${CMAKE_CURRENT_BINARY_DIR}/wind3d-verify-${devices}.bin")
        execute_process(
            COMMAND "${WIND3D_TASK}" --grid 80 --gpus ${devices}
                    --steps 8 --warmup 2 --validate ${export_options} --dump "${snapshot}"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE output ERROR_VARIABLE errors
            TIMEOUT 300)
        if (NOT result EQUAL 0)
            message(FATAL_ERROR "${devices}-GPU validation failed: ${result}\n${output}\n${errors}")
        endif ()
        file(SHA256 "${snapshot}" checksum)
        file(REMOVE "${snapshot}")
        if (devices EQUAL 1 AND mode STREQUAL full)
            set(reference "${checksum}")
        elseif (NOT checksum STREQUAL reference)
            message(FATAL_ERROR "${devices}-GPU field differs from the single-GPU reference")
        endif ()
    endforeach ()
endforeach ()
message(STATUS "1/2/4/8-GPU fields match bit for bit in full/final export modes; geometry, projection and lifecycle checks passed")
