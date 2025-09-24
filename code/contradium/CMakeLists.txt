
add_library(Contradium)

target_sources(Contradium PRIVATE contradium.cpp)

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    # check whether ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt exists
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
        set(CONTRADIUM_SUBLIB_NAME contradium_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

target_link_libraries(Contradium PUBLIC ${CONTRADIUM_LIBS})
