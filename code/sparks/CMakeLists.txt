
add_library(Sparks)

target_sources(Sparks PRIVATE sparks.cpp)

set(LONGMARCH_SPARKS_SHADERS ${CMAKE_CURRENT_LIST_DIR}/shaders)


file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
# check whether ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt exists
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
        set(SPARKS_SUBLIB_NAME sparks_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

target_link_libraries(Sparks PUBLIC ${SPARKS_LIBS})
