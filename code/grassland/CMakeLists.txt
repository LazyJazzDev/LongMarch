if (LONGMARCH_CUDA_ENABLED)
    file(GLOB SOURCES "*.cpp" "*.h" "*.cu" "*.cuh")
else ()
    file(GLOB SOURCES "*.cpp" "*.h")
endif ()

add_library(CaoDi ${SOURCES})

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        set(CAO_DI_SUBLIB_NAME cao_di_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

message(STATUS "CAO_DI_LIBS: ${CAO_DI_LIBS}")

target_link_libraries(CaoDi PUBLIC ${CAO_DI_LIBS})
