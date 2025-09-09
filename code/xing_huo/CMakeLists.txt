
add_library(XingHuo)

target_sources(XingHuo PRIVATE xing_huo.cpp)

set(CHANGZHENG_XING_HUO_SHADERS ${CMAKE_CURRENT_LIST_DIR}/shaders)


file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
# check whether ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt exists
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
        set(XING_HUO_SUBLIB_NAME xing_huo_${subdir})
        add_subdirectory(${subdir})
    endif ()
endforeach ()

target_link_libraries(XingHuo PUBLIC ${XING_HUO_LIBS})
