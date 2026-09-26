# Source-only package: target libraries are compiled by the mobile SDK.
vcpkg_from_github(OUT_SOURCE_PATH MIKK_SOURCE REPO mmikk/MikkTSpace
    REF 3e895b49d05ea07e4c2133156cfa94369e19e409
    SHA512 3ca433bd4efd0e048138f9efc5ba9021e4f3f78a535ea48733088ba5f43e60aad7f840f00e0597a0c053cda4776177bf6deb14cecf4d172b9b68acf00d5a1ca7)
vcpkg_from_gitlab(GITLAB_URL https://gitlab.freedesktop.org/
    OUT_SOURCE_PATH FT_SOURCE REPO freetype/freetype REF VER-2-13-3
    SHA512 fccfaa15eb79a105981bf634df34ac9ddf1c53550ec0b334903a1b21f9f8bf5eb2b3f9476e554afa112a0fca58ec85ab212d674dfd853670efec876bacbe8a53)
set(destination "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(INSTALL "${MIKK_SOURCE}/mikktspace.h" "${MIKK_SOURCE}/mikktspace.c" DESTINATION "${destination}/mikktspace")
file(INSTALL "${FT_SOURCE}/" DESTINATION "${destination}/freetype")
set(VCPKG_POLICY_EMPTY_PACKAGE enabled)
vcpkg_install_copyright(FILE_LIST "${MIKK_SOURCE}/mikktspace.h" "${FT_SOURCE}/docs/FTL.TXT" "${FT_SOURCE}/docs/GPLv2.TXT")
