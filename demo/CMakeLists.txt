# Enumerate all subdirectories of current directory
file(GLOB subdirs CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach (subdir ${subdirs})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        set(DEMO_NAME demo_${subdir})
        add_subdirectory(${subdir})
    endif()
endforeach()
