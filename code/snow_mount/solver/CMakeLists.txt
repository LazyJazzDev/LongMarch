file(GLOB_RECURSE SOURCES "*.cpp")

if (LONGMARCH_CUDA_ENABLED)
    # mark *.cpp as CUDA sources
    foreach (source IN LISTS SOURCES)
        if (source MATCHES ".*\\.cpp$")
            set_property(SOURCE ${source} PROPERTY LANGUAGE CUDA)
        endif ()
    endforeach ()
endif ()

add_library(${XUE_SHAN_SUBLIB_NAME} ${SOURCES})

list(APPEND XUE_SHAN_LIBS ${XUE_SHAN_SUBLIB_NAME})
set(XUE_SHAN_LIBS ${XUE_SHAN_LIBS} PARENT_SCOPE)

target_include_directories(${XUE_SHAN_SUBLIB_NAME} PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(${XUE_SHAN_SUBLIB_NAME} PUBLIC CaoDi)

PACK_SHADER_CODE(${XUE_SHAN_SUBLIB_NAME})
