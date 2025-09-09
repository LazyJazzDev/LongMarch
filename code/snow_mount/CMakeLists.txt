file(GLOB SOURCES "*.cpp")

add_library(XueShan ${SOURCES})

file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        set(XUE_SHAN_SUBLIB_NAME xue_shan_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

message(STATUS "XUE_SHAN_LIBS: ${XUE_SHAN_LIBS}")

target_link_libraries(XueShan PUBLIC ${XUE_SHAN_LIBS})
