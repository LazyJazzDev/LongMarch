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
    # Use a matching pair of runtime DLLs from the Windows SDK. Their architecture
    # must match the application loading Slang, not the build host.
    if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
        set(_dxc_arch arm64)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_dxc_arch x64)
    else()
        set(_dxc_arch x86)
    endif()
    set(LONGMARCH_DXC_RUNTIME_DIR "" CACHE PATH
        "Windows SDK directory containing dxcompiler.dll and dxil.dll")
    if(LONGMARCH_DXC_RUNTIME_DIR)
        set(_dxc_bin "${LONGMARCH_DXC_RUNTIME_DIR}")
    else()
        get_filename_component(_windows_sdk_root
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]"
            ABSOLUTE)
        if(DEFINED ENV{WindowsSdkDir})
            file(TO_CMAKE_PATH "$ENV{WindowsSdkDir}" _windows_sdk_root)
        endif()
        file(GLOB _sdk_bins LIST_DIRECTORIES TRUE "${_windows_sdk_root}/bin/10.*")
        list(SORT _sdk_bins COMPARE NATURAL ORDER DESCENDING)
        # Prefer the SDK selected by the generator or developer command prompt.
        if(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
            list(PREPEND _sdk_bins
                "${_windows_sdk_root}/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
        elseif(DEFINED ENV{WindowsSDKVersion})
            file(TO_CMAKE_PATH "$ENV{WindowsSDKVersion}" _sdk_version)
            string(REGEX REPLACE "/$" "" _sdk_version "${_sdk_version}")
            list(PREPEND _sdk_bins "${_windows_sdk_root}/bin/${_sdk_version}")
        endif()
        unset(_dxc_bin)
        foreach(_sdk_bin IN LISTS _sdk_bins)
            if(EXISTS "${_sdk_bin}/${_dxc_arch}/dxcompiler.dll" AND
               EXISTS "${_sdk_bin}/${_dxc_arch}/dxil.dll")
                set(_dxc_bin "${_sdk_bin}/${_dxc_arch}")
                break()
            endif()
        endforeach()
    endif()
    if(NOT EXISTS "${_dxc_bin}/dxcompiler.dll" OR NOT EXISTS "${_dxc_bin}/dxil.dll")
        message(FATAL_ERROR
            "Windows SDK DXC runtime (${_dxc_arch}) was not found. Install a recent "
            "Windows SDK or set LONGMARCH_DXC_RUNTIME_DIR to its bin/<version>/${_dxc_arch} directory.")
    endif()
    message(STATUS "Windows SDK DXC runtime: ${_dxc_bin}")
    get_target_property(_slang_location slang::slang IMPORTED_LOCATION_RELEASE)
    if(NOT _slang_location)
        get_target_property(_slang_location slang::slang IMPORTED_LOCATION)
    endif()
    get_filename_component(_slang_bin "${_slang_location}" DIRECTORY)
    # vcpkg's bin directory is shared with unrelated dependencies.
    file(GLOB LONGMARCH_SLANG_RUNTIME_DLLS "${_slang_bin}/slang*.dll")
    foreach(_name dxcompiler.dll dxil.dll)
        list(APPEND LONGMARCH_SLANG_RUNTIME_DLLS "${_dxc_bin}/${_name}")
    endforeach()
endif()
