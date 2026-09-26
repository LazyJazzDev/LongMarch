# Dependencies are supplied by vcpkg or an explicitly selected external SDK.
# Never download a compiler during CMake package discovery.
set(LONGMARCH_MIN_SLANG_VERSION "2026.18.1")
# Slang's config version file restricts the calendar year as a major version.
# Compare the lower bound ourselves so a compatible SDK from a later year is
# not rejected solely because its year differs.
find_package(slang CONFIG QUIET)
if(NOT slang_FOUND OR NOT DEFINED slang_VERSION OR slang_VERSION VERSION_LESS LONGMARCH_MIN_SLANG_VERSION)
    message(FATAL_ERROR
        "Slang ${LONGMARCH_MIN_SLANG_VERSION}+ is required. Install the default vcpkg "
        "slang feature, or set slang_DIR to an external SDK's CMake package directory. "
        "For an external SDK, use -DVCPKG_MANIFEST_NO_DEFAULT_FEATURES=ON. "
        "CMake will not download Slang.")
endif()
message(STATUS "Slang SDK: ${slang_DIR}")

if(WIN32)
    # Official Slang releases do not bundle the downstream DXIL compiler.
    # vcpkg provides it as a runtime dependency; no DXC API is linked by LongMarch.
    find_package(directx-dxc CONFIG REQUIRED)
    get_filename_component(_dxc_bin "${DIRECTX_DXC_TOOL}" DIRECTORY)
    get_target_property(_slang_location slang::slang IMPORTED_LOCATION_RELEASE)
    if(NOT _slang_location)
        get_target_property(_slang_location slang::slang IMPORTED_LOCATION)
    endif()
    get_filename_component(_slang_bin "${_slang_location}" DIRECTORY)
    # vcpkg's bin directory is shared with unrelated dependencies.
    file(GLOB LONGMARCH_SLANG_RUNTIME_DLLS "${_slang_bin}/slang*.dll")
    foreach(_name dxcompiler.dll dxil.dll)
        if(NOT EXISTS "${_dxc_bin}/${_name}")
            message(FATAL_ERROR "Slang DXIL target requires ${_dxc_bin}/${_name}")
        endif()
        list(APPEND LONGMARCH_SLANG_RUNTIME_DLLS "${_dxc_bin}/${_name}")
    endforeach()
endif()
