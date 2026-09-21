add_library(sparkium_cuda_tracing raytracing.cpp)

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    # check whether ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt exists
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
        set(SPARKIUM_CUDA_TRACING_SUBLIB_NAME sparkium_cuda_tracing_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

target_link_libraries(sparkium_cuda_tracing PUBLIC ${SPARKIUM_CUDA_TRACING_LIBS} Grassland Snowberg)
