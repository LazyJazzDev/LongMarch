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


# Construct platform-specific exclude regexes
set(_pre_excludes_common
        [=[api-ms-]=]
        [=[ext-ms-]=]
)

set(_post_excludes_common
        [=[.*system32\/.*\.dll]=]           # Windows system32
)

# Linux specific excludes
set(_linux_excludes
        [[libc\.so\..*]] [[libgcc_s\.so\..*]] [[libm\.so\..*]] [[libstdc\+\+\.so\..*]]
        [[python.*]]                        # libpython on Linux
        [=[^\/(lib|usr\/lib|usr\/local\/lib)]=]
)

# macOS specific excludes
# - Exclude system libraries and Apple frameworks
# - Exclude system Python framework and libpython
set(_macos_pre_excludes
        # none specific needed here, keep for symmetry
)

set(_macos_post_excludes
        [=[^/System/]=]
        [=[^/usr/lib/]=]
        [=[^/Library/Apple/]=]
        # Exclude system frameworks
        [=[^/System/Library/Frameworks/]=]
        [=[Python\.framework]=]
        [[libpython[0-9]+\.[0-9]+\.dylib]]
)

# Windows specific excludes (keep your existing)
set(_windows_excludes
        [[kernel32\.dll]]
)

# Compose the final exclude lists by platform
set(_pre_excludes ${_pre_excludes_common})
set(_post_excludes ${_post_excludes_common})

if(WIN32)
    list(APPEND _pre_excludes ${_windows_excludes})
elseif(APPLE)
    list(APPEND _pre_excludes ${_macos_pre_excludes})
    list(APPEND _post_excludes ${_macos_post_excludes})
else() # Linux and other UNIX
    list(APPEND _pre_excludes ${_linux_excludes})
endif()

install(
        RUNTIME_DEPENDENCY_SET long_march_deps
        DESTINATION ${Python3_SITEARCH}
        PRE_EXCLUDE_REGEXES
        ${_pre_excludes}
        POST_EXCLUDE_REGEXES
        ${_post_excludes}
        DIRECTORIES "$<TARGET_FILE_DIR:long_march>"
        "${DXC_DLL_DIR}"
        "${Python3_RUNTIME_LIBRARY_DIRS}"
        COMPONENT python_package
)

if (APPLE)
    set_target_properties(long_march PROPERTIES
            INSTALL_RPATH "@loader_path;${Python3_SITEARCH}"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif ()

message(STATUS "DXC_DLL_DIR: ${DXC_DLL_DIR}")
