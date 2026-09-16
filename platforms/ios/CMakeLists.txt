cmake_minimum_required(VERSION 3.25)
project(SparkiumMobile LANGUAGES C CXX OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)
option(SPARKIUM_PREPARE "Build macOS shader preparation tool (requires DXC and SPIRV-Cross)" OFF)
option(SPARKIUM_APP "Build native SwiftUI application" ON)
set(SPARKIUM_HEADERS "" CACHE PATH "Portable dependency headers (vcpkg installed/<triplet>/include)")
set(SPARKIUM_RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../out/ios/Resources" CACHE PATH "Prepared resource bundle")
if(NOT APPLE)
  message(FATAL_ERROR "SparkiumMobile requires an Apple SDK")
endif()
if(SPARKIUM_PREPARE AND CMAKE_SYSTEM_NAME STREQUAL "iOS")
  message(FATAL_ERROR "Run shader preparation on macOS")
endif()
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
include(FetchContent)
FetchContent_Declare(metal_cpp URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip
  URL_HASH SHA256=0433df1e0ab13c2b0becbd78665071e3fa28381e9714a3fce28a497892b8a184 DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(mikktspace URL https://github.com/mmikk/MikkTSpace/archive/3e895b49d05ea07e4c2133156cfa94369e19e409.tar.gz DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(metal_cpp mikktspace)
if(NOT EXISTS "${SPARKIUM_HEADERS}/fmt/format.h" OR NOT EXISTS "${SPARKIUM_HEADERS}/eigen3/Eigen/Core")
  message(FATAL_ERROR "Set SPARKIUM_HEADERS to headers installed using platforms/ios/vcpkg.json")
endif()
file(GLOB UTIL "${ROOT}/code/grassland/util/*.cpp")
file(GLOB GRAPHICS "${ROOT}/code/grassland/graphics/*.cpp")
list(FILTER GRAPHICS EXCLUDE REGEX "/window.cpp$")
file(GLOB METAL "${ROOT}/code/grassland/graphics/backend/metal/*.cpp")
file(GLOB_RECURSE SPARKIUM "${ROOT}/code/sparkium/*.cpp")
add_library(sparkium_mobile STATIC ${UTIL} ${GRAPHICS} ${METAL} ${SPARKIUM}
  "${ROOT}/code/grassland/math/math_mesh.cxx" "${mikktspace_SOURCE_DIR}/mikktspace.c" RenderSession.cpp)
target_include_directories(sparkium_mobile PUBLIC "${ROOT}/code" "${SPARKIUM_HEADERS}"
  "${SPARKIUM_HEADERS}/eigen3" "${SPARKIUM_HEADERS}/stb" "${metal_cpp_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
  "${mikktspace_SOURCE_DIR}")
target_compile_definitions(sparkium_mobile PUBLIC LONGMARCH_METAL_ENABLED LONGMARCH_HEADLESS
  FMT_HEADER_ONLY NDEBUG LONGMARCH_ASSETS_DIR="")
target_link_libraries(sparkium_mobile PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
# Generate embedded HLSL on the host, without executing a cross-compiled helper.
foreach(pair "grassland/graphics" "sparkium/shaders")
  string(REPLACE "/" "_" name "${pair}")
  set(generated "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  file(MAKE_DIRECTORY "${generated}")
  file(GLOB_RECURSE shaders CONFIGURE_DEPENDS "${ROOT}/code/${pair}/*.hlsl" "${ROOT}/code/${pair}/*.hlsli")
  add_custom_command(OUTPUT "${generated}/built_in_shaders.inl"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.py"
      "${ROOT}/code/${pair}" "${generated}/built_in_shaders.inl"
    DEPENDS ${shaders} embed_shaders.py VERBATIM)
  target_sources(sparkium_mobile PRIVATE "${generated}/built_in_shaders.inl")
endforeach()
set_source_files_properties("${ROOT}/code/grassland/graphics/shader.cpp" PROPERTIES
  INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/grassland_graphics")
# Sparkium's generated include is used by core/core_util.cpp.
file(GLOB_RECURSE spark_sources "${ROOT}/code/sparkium/*.cpp")
set_source_files_properties(${spark_sources} PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/sparkium_shaders")
if(SPARKIUM_PREPARE)
  find_path(DXC_INCLUDE_DIR dxc/dxcapi.h HINTS /usr/local/include REQUIRED)
  find_library(DXC_LIBRARY dxcompiler HINTS /usr/local/lib REQUIRED)
  find_path(SPIRV_CROSS_INCLUDE_DIR spirv_cross/spirv_msl.hpp HINTS /usr/local/include REQUIRED)
  target_include_directories(sparkium_mobile PRIVATE "${DXC_INCLUDE_DIR}" "${SPIRV_CROSS_INCLUDE_DIR}")
  target_link_libraries(sparkium_mobile PUBLIC "${DXC_LIBRARY}")
  foreach(part msl glsl core)
    find_library(SPIRV_CROSS_${part} spirv-cross-${part} HINTS /usr/local/lib REQUIRED)
    target_link_libraries(sparkium_mobile PUBLIC "${SPIRV_CROSS_${part}}")
  endforeach()
else()
  target_compile_definitions(sparkium_mobile PRIVATE LONGMARCH_OFFLINE_SHADERS)
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
  add_executable(sparkium_mobile_check check.cpp)
  target_link_libraries(sparkium_mobile_check PRIVATE sparkium_mobile)
endif()
if(SPARKIUM_APP)
  enable_language(Swift)
  if(NOT EXISTS "${SPARKIUM_RESOURCES}/catalog.json")
    message(FATAL_ERROR "Prepare resources first; see README.md")
  endif()
  # A root folder literally named Resources makes CFBundle interpret an iOS app
  # as a legacy macOS bundle and overlook its root Info.plist.
  set(bundle_resources "${CMAKE_CURRENT_BINARY_DIR}/SparkiumResources")
  file(CREATE_LINK "${SPARKIUM_RESOURCES}" "${bundle_resources}" SYMBOLIC)
  add_executable(Sparkium MACOSX_BUNDLE SparkiumApp.swift SparkiumBridge.mm SparkiumBridge.h
    "${bundle_resources}" Assets.xcassets)
  set_source_files_properties("${bundle_resources}" Assets.xcassets PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
  target_link_libraries(Sparkium PRIVATE sparkium_mobile "-framework SwiftUI" "-framework CoreGraphics")
  target_compile_options(Sparkium PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
  set_target_properties(Sparkium PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "dev.lazyjazz.sparkium"
    XCODE_ATTRIBUTE_ASSETCATALOG_COMPILER_APPICON_NAME "AppIcon"
    XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/SparkiumBridge.h"
    XCODE_ATTRIBUTE_SWIFT_VERSION "5.0"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
    XCODE_ATTRIBUTE_ENABLE_USER_SCRIPT_SANDBOXING NO)
endif()
