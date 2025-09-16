file(GLOB SOURCES "*.cpp" "*.h")

add_library(${SPARKS_SUBLIB_NAME} ${SOURCES})

target_link_libraries(${SPARKS_SUBLIB_NAME} PUBLIC Grassland Snowberg)
list(APPEND SPARKS_LIBS ${SPARKS_SUBLIB_NAME})
set(SPARKS_LIBS ${SPARKS_LIBS} PARENT_SCOPE)
