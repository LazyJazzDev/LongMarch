vcpkg_download_distfile(ARCHIVE
    URLS "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
    FILENAME "metal-cpp_macOS15_iOS18.zip"
    SHA512 5b24953eb70f128062faca8f0a0130fcb6e0b837427c75d32930f6a4146b87413b5c4195cffbbbf13024f878ea2883817113df54550885daa90238ab372ffe4c)
vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${ARCHIVE}")
file(INSTALL "${SOURCE_PATH}/Foundation" "${SOURCE_PATH}/Metal"
    "${SOURCE_PATH}/QuartzCore" "${SOURCE_PATH}/MetalFX"
    DESTINATION "${CURRENT_PACKAGES_DIR}/include/metal-cpp")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
