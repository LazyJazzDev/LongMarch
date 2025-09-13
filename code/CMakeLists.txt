list(APPEND LONGMARCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})
list(APPEND LONGMARCH_INCLUDE_DIR ${INC_DIR})
set(LONGMARCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)

add_library(LongMarch long_march.cpp)

add_subdirectory(grassland)
add_subdirectory(snowberg)
add_subdirectory(sparks)

target_include_directories(LongMarch PUBLIC ${LONGMARCH_INCLUDE_DIR})
target_link_libraries(LongMarch PUBLIC Grassland Snowberg Sparks)

pybind11_add_module(long_march long_march_pybind.cpp)

target_link_libraries(long_march PUBLIC LongMarch)

# copy all related dll files to the destination too.
install(TARGETS long_march
        RUNTIME_DEPENDENCY_SET long_march_deps
        LIBRARY DESTINATION ${Python3_SITEARCH}
        COMPONENT python_package)

install(
        RUNTIME_DEPENDENCY_SET long_march_deps
        DESTINATION ${Python3_SITEARCH}
        PRE_EXCLUDE_REGEXES
        [=[api-ms-]=]
        [=[ext-ms-]=]
        [[kernel32\.dll]]
        [[libc\.so\..*]] [[libgcc_s\.so\..*]] [[libm\.so\..*]] [[libstdc\+\+\.so\..*]]
        [[python*]]
        POST_EXCLUDE_REGEXES
        [=[.*system32\/.*\.dll]=]
        [=[^\/(lib|usr\/lib|usr\/local\/lib)]=]
        DIRECTORIES "$<TARGET_FILE_DIR:long_march>"
        "${DXC_DLL_DIR}"
        COMPONENT python_package
)
