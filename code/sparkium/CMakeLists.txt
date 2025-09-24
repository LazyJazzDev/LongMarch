
add_library(Sparkium)

target_sources(Sparkium PRIVATE sparkium.cpp)

set(LONGMARCH_SPARKIUM_SHADERS ${CMAKE_CURRENT_LIST_DIR}/shaders)

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
# check whether ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt exists
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
        set(SPARKIUM_SUBLIB_NAME sparkium_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

target_link_libraries(Sparkium PUBLIC ${SPARKIUM_LIBS})
