file(GLOB SOURCES CONFIGURE_DEPENDS "*.h" "*.cpp")

add_library(${GRASSLAND_SUBLIB_NAME} ${SOURCES})

list(APPEND GRASSLAND_LIBS ${GRASSLAND_SUBLIB_NAME})
set(GRASSLAND_LIBS ${GRASSLAND_LIBS} PARENT_SCOPE)

target_include_directories(${GRASSLAND_SUBLIB_NAME} PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(${GRASSLAND_SUBLIB_NAME} PRIVATE grassland_util slang::slang PUBLIC ${GLFW3_LIB_NAME})

# Display brightness queries are shared by all Windows rendering backends.
if(WIN32)
    target_link_libraries(${GRASSLAND_SUBLIB_NAME} PRIVATE dxgi)
endif()

add_subdirectory(backend)

add_subdirectory(rhix)
