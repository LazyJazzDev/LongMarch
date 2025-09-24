add_library(Practium)

target_sources(Practium PRIVATE practium.cpp)

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    # check whether ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt exists
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
        set(PRACTIUM_SUBLIB_NAME practium_${subdir})
        add_subdirectory(${subdir})
        message(STATUS "Added practium subdirectory: ${subdir}")
    endif ()
endforeach ()

target_link_libraries(Practium PUBLIC ${PRACTIUM_LIBS})

message(STATUS "PRACTIUM_LIBS: ${PRACTIUM_LIBS}")
