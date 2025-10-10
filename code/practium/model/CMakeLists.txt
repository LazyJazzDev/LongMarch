file(GLOB SOURCES "*.cpp" "*.h")

add_library(${PRACTIUM_SUBLIB_NAME} ${SOURCES})

target_link_libraries(${PRACTIUM_SUBLIB_NAME} PUBLIC Grassland Snowberg Sparkium Contradium practium_core)
list(APPEND PRACTIUM_LIBS ${PRACTIUM_SUBLIB_NAME})
set(PRACTIUM_LIBS ${PRACTIUM_LIBS} PARENT_SCOPE)
