cmake_minimum_required(VERSION 3.25)
project(LongMarch LANGUAGES C CXX OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)
option(SPARKIUM_PREPARE "Build macOS shader preparation tool (requires Slang and SPIRV-Cross)" OFF)
option(SPARKIUM_APP "Build native SwiftUI application" ON)
set(SPARKIUM_HEADERS "" CACHE PATH "Portable dependency headers (vcpkg installed/<triplet>/include)")
set(SPARKIUM_RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../out/ios/Resources" CACHE PATH "Prepared resource bundle")
if(NOT APPLE)
  message(FATAL_ERROR "LongMarch requires an Apple SDK")
endif()
if(SPARKIUM_PREPARE AND CMAKE_SYSTEM_NAME STREQUAL "iOS")
  message(FATAL_ERROR "Run shader preparation on macOS")
endif()
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
include("${ROOT}/platforms/mobile/Dependencies.cmake")
if(NOT SPARKIUM_HEADERS)
  set(SPARKIUM_HEADERS "${LONGMARCH_MOBILE_HEADERS}")
endif()
if(NOT EXISTS "${SPARKIUM_HEADERS}/fmt/format.h" OR NOT EXISTS "${SPARKIUM_HEADERS}/eigen3/Eigen/Core")
  message(FATAL_ERROR "Set SPARKIUM_HEADERS to headers installed using platforms/ios/vcpkg.json")
endif()
file(GLOB UTIL "${ROOT}/code/grassland/util/*.cpp")
file(GLOB GRAPHICS "${ROOT}/code/grassland/graphics/*.cpp")
list(FILTER GRAPHICS EXCLUDE REGEX "/window.cpp$")
file(GLOB METAL "${ROOT}/code/grassland/graphics/backend/metal/*.cpp")
file(GLOB_RECURSE SPARKIUM "${ROOT}/code/sparkium/*.cpp")
add_library(sparkium_mobile STATIC ${UTIL} ${GRAPHICS} ${METAL} ${SPARKIUM}
  "${ROOT}/code/grassland/math/math_mesh.cxx" "${mikktspace_SOURCE_DIR}/mikktspace.c" RenderSession.cpp
  "${ROOT}/demo/gol/game_of_life_lib/game_of_life_lib.cpp"
  "${ROOT}/demo/2048/2048_lib/2048_lib.cpp" "${ROOT}/demo/2048/game_rules.cpp" "${ROOT}/demo/2048/ai_player.cpp")
target_include_directories(sparkium_mobile PUBLIC "${ROOT}/code" "${SPARKIUM_HEADERS}"
  "${SPARKIUM_HEADERS}/eigen3" "${SPARKIUM_HEADERS}/stb" "${metal_cpp_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
  "${mikktspace_SOURCE_DIR}" "${ROOT}" "${ROOT}/demo/2048/2048_lib")
target_compile_definitions(sparkium_mobile PUBLIC LONGMARCH_METAL_ENABLED LONGMARCH_HEADLESS
  FMT_HEADER_ONLY NDEBUG "$<$<NOT:$<COMPILE_LANGUAGE:Swift>>:LONGMARCH_ASSETS_DIR=\"\">")
target_link_libraries(sparkium_mobile PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")
add_library(mobile_demos STATIC demos/DemoSession.cpp games/DesktopGameSession.cpp games/image_write.cpp)
target_link_libraries(mobile_demos PUBLIC sparkium_mobile ios_gol_ui ios_2048_ui)
target_include_directories(mobile_demos PRIVATE "${ROOT}/demo/gol/game_of_life_lib" "${ROOT}/demo/2048" "${ROOT}/demo/gol")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
# Generate embedded Slang on the host, without executing a cross-compiled helper.
foreach(pair "grassland/graphics" "sparkium/shaders")
  string(REPLACE "/" "_" name "${pair}")
  set(generated "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  file(MAKE_DIRECTORY "${generated}")
  file(GLOB_RECURSE shaders CONFIGURE_DEPENDS "${ROOT}/code/${pair}/*.slang")
  add_custom_command(OUTPUT "${generated}/built_in_shaders.inl"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.py"
      "${ROOT}/code/${pair}" "${generated}/built_in_shaders.inl"
    DEPENDS ${shaders} embed_shaders.py VERBATIM)
  target_sources(sparkium_mobile PRIVATE "${generated}/built_in_shaders.inl")
endforeach()
file(GLOB_RECURSE demo_shaders CONFIGURE_DEPENDS "${ROOT}/demo/*.slang")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/demo_shaders.inl"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.py"
    "${ROOT}/demo" "${CMAKE_CURRENT_BINARY_DIR}/demo_shaders.inl"
  DEPENDS ${demo_shaders} embed_shaders.py VERBATIM)
target_sources(mobile_demos PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/demo_shaders.inl")
set_source_files_properties(demos/DemoSession.cpp PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}")
set_source_files_properties("${ROOT}/code/grassland/graphics/shader.cpp" PROPERTIES
  INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/grassland_graphics")
# Sparkium's generated include is used by core/core_util.cpp.
file(GLOB_RECURSE spark_sources "${ROOT}/code/sparkium/*.cpp")
set_source_files_properties(${spark_sources} PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/sparkium_shaders")
if(SPARKIUM_PREPARE)
  longmarch_mobile_compiler(sparkium_mobile)
  # Same exported SPIRV-Cross targets as the desktop Metal backend (cmake/Metal.cmake).
  foreach(part core glsl msl)
    find_package(spirv_cross_${part} CONFIG REQUIRED HINTS ${_mobile_prefixes})
  endforeach()
  target_link_libraries(sparkium_mobile PUBLIC spirv-cross-msl spirv-cross-glsl spirv-cross-core)
else()
  target_compile_definitions(sparkium_mobile PRIVATE LONGMARCH_OFFLINE_SHADERS)
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
  if(SPARKIUM_PREPARE)
    add_executable(harmony_fallback_prepare "${ROOT}/platforms/harmonyos/native/scene_check.cpp")
    target_link_libraries(harmony_fallback_prepare PRIVATE sparkium_mobile)
    # WGSL for the browser apps (platforms/web), from the same embedded Slang.
    add_executable(web_shader_prepare "${ROOT}/platforms/web/prepare_shaders.cpp" "${CMAKE_CURRENT_BINARY_DIR}/demo_shaders.inl")
    target_include_directories(web_shader_prepare PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(web_shader_prepare PRIVATE sparkium_mobile)
  endif()
  add_executable(mobile_display_check tests/display_image_check.cpp)
  target_include_directories(mobile_display_check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(mobile_display_check PRIVATE "-framework CoreGraphics" "-framework CoreFoundation")
  add_executable(sparkium_mobile_check check.cpp)
  target_link_libraries(sparkium_mobile_check PRIVATE sparkium_mobile)
  add_executable(gol_benchmark tests/gol_benchmark.cpp)
  target_link_libraries(gol_benchmark PRIVATE mobile_demos)
  add_executable(mobile_games_check tests/games_check.cpp)
  target_link_libraries(mobile_games_check PRIVATE mobile_demos)
  add_executable(mobile_demo_check demos/check.cpp)
  target_link_libraries(mobile_demo_check PRIVATE mobile_demos)
endif()
set(GAME_OF_LIFE_RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../out/ios/GameOfLifeResources" CACHE PATH
  "Prepared resources of the standalone Game of Life app (prepare_game_resources.py)")

# Apps can also be defined by a project that adds this directory with
# add_subdirectory (for example LongMarch as a Git submodule): it calls
# longmarch_ios_enable_swift() and then longmarch_ios_app().
set(LONGMARCH_IOS_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "LongMarch iOS host sources")

macro(longmarch_ios_enable_swift)
  if(CMAKE_GENERATOR STREQUAL "Ninja" AND CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_longmarch_swift_target "arm64-apple-ios${CMAKE_OSX_DEPLOYMENT_TARGET}")
    string(TOLOWER "${CMAKE_OSX_SYSROOT}" _longmarch_sysroot)
    if(_longmarch_sysroot MATCHES "simulator")
      string(APPEND _longmarch_swift_target "-simulator")
    endif()
    set(CMAKE_Swift_FLAGS "${CMAKE_Swift_FLAGS} -target ${_longmarch_swift_target}")
  endif()
  enable_language(Swift)
endmacro()

# One SwiftUI app bundle. RESOURCES is a prepared folder copied into the bundle
# as FOLDER; STRINGS are <language>.lproj/<name>.strings files. SOURCES and
# STRINGS are relative to the calling directory; the shared games UI is
# ${LONGMARCH_IOS_DIR}/games/GamesView.swift with its Localizable.strings.
function(longmarch_ios_app target bundle_id info_plist assets resources folder)
  cmake_parse_arguments(PARSE_ARGV 6 app "" "" "SOURCES;STRINGS")
  # A root folder literally named Resources makes CFBundle interpret an iOS app
  # as a legacy macOS bundle and overlook its root Info.plist.
  set(bundle_resources "${CMAKE_CURRENT_BINARY_DIR}/${target}-bundle/${folder}")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${target}-bundle")
  file(CREATE_LINK "${resources}" "${bundle_resources}" SYMBOLIC)
  set(host "${LONGMARCH_IOS_DIR}")
  if(NOT Python3_EXECUTABLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
  endif()
  add_executable(${target} MACOSX_BUNDLE ${app_SOURCES}
    "${host}/demos/DemoBridge.mm" "${host}/demos/DemoBridge.h" "${host}/SparkiumBridge.mm"
    "${host}/SparkiumBridge.h" "${host}/RenderQueue.h" "${bundle_resources}" ${assets} ${app_STRINGS})
  set_source_files_properties("${bundle_resources}" ${assets} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
  if(assets MATCHES "\\.icon$")
    # An Icon Composer icon: layered Liquid Glass on iOS 26, flat renderings before.
    set_source_files_properties(${assets} PROPERTIES XCODE_EXPLICIT_FILE_TYPE folder.iconcomposer.icon)
  endif()
  foreach(strings ${app_STRINGS})
    get_filename_component(language "${strings}" DIRECTORY)
    get_filename_component(language "${language}" NAME)
    set_source_files_properties(${strings} PROPERTIES MACOSX_PACKAGE_LOCATION Resources/${language})
  endforeach()
  # Folder resources otherwise copy only at configure time; shader preparation can
  # add nested cache files without changing the folder timestamp seen by Ninja.
  add_custom_target(${target}_resources
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${resources}" "$<TARGET_BUNDLE_DIR:${target}>/${folder}"
    VERBATIM)
  add_dependencies(${target} ${target}_resources)
  target_link_libraries(${target} PRIVATE mobile_demos "-framework SwiftUI" "-framework CoreGraphics" "-framework MetalKit")
  target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
  set_target_properties(${target} PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST "${info_plist}"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${bundle_id}"
    XCODE_ATTRIBUTE_ASSETCATALOG_COMPILER_APPICON_NAME "AppIcon"
    XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${host}/SparkiumBridge.h"
    XCODE_ATTRIBUTE_SWIFT_VERSION "5.0"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
    XCODE_ATTRIBUTE_ENABLE_USER_SCRIPT_SANDBOXING NO
    # Xcode archives only install the app; CMake skips installation by default.
    XCODE_ATTRIBUTE_SKIP_INSTALL NO
    XCODE_ATTRIBUTE_INSTALL_PATH "$(LOCAL_APPS_DIR)")
  if(CMAKE_GENERATOR STREQUAL "Ninja")
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:Swift>:-import-objc-header;${host}/SparkiumBridge.h>")
    target_link_options(${target} PRIVATE -sdk "${CMAKE_OSX_SYSROOT}")
    set_target_properties(${target} PROPERTIES MACOSX_BUNDLE_GUI_IDENTIFIER "${bundle_id}")
    set(ios_platform iphoneos)
    string(TOLOWER "${CMAKE_OSX_SYSROOT}" ios_sysroot_lower)
    if(ios_sysroot_lower MATCHES "simulator")
      set(ios_platform iphonesimulator)
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND xcrun actool "${assets}" --compile "$<TARGET_BUNDLE_DIR:${target}>"
        --platform ${ios_platform} --minimum-deployment-target ${CMAKE_OSX_DEPLOYMENT_TARGET}
        --app-icon AppIcon --target-device iphone --target-device ipad
        --output-partial-info-plist "${CMAKE_CURRENT_BINARY_DIR}/${target}-bundle/AppIcon.plist"
      COMMAND "${Python3_EXECUTABLE}" "${host}/finish_bundle.py"
        "$<TARGET_BUNDLE_DIR:${target}>" "${CMAKE_CURRENT_BINARY_DIR}/${target}-bundle/AppIcon.plist" ${target} ${bundle_id}
      VERBATIM)
  endif()
endfunction()

if(SPARKIUM_APP)
  longmarch_ios_enable_swift()
  if(NOT EXISTS "${SPARKIUM_RESOURCES}/catalog.json")
    message(FATAL_ERROR "Prepare resources first; see README.md")
  endif()
  set(game_strings en.lproj/Localizable.strings zh-Hans.lproj/Localizable.strings)
  longmarch_ios_app(LongMarch dev.lazyjazz.longmarch "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist"
    "${CMAKE_CURRENT_SOURCE_DIR}/Assets.xcassets" "${SPARKIUM_RESOURCES}" SparkiumResources
    SOURCES SparkiumApp.swift DemosApp.swift demos/DemoView.swift games/GamesView.swift
    STRINGS ${game_strings})
  if(EXISTS "${GAME_OF_LIFE_RESOURCES}/Patterns/library.json")
    longmarch_ios_app(GameOfLife net.lazyjazz.gameoflife "${CMAKE_CURRENT_SOURCE_DIR}/gol/Info.plist"
      "${CMAKE_CURRENT_SOURCE_DIR}/gol/AppIcon.icon" "${GAME_OF_LIFE_RESOURCES}" GameResources
      SOURCES gol/GameOfLifeApp.swift games/GamesView.swift
      STRINGS ${game_strings} gol/en.lproj/InfoPlist.strings gol/zh-Hans.lproj/InfoPlist.strings)
  else()
    message(STATUS "Game of Life app skipped: prepare GAME_OF_LIFE_RESOURCES first; see README.md")
  endif()
endif()

# Compile the actual desktop UI, geometry, animation and Slang for native hosts.
# Game namespaces allow both original applications in one executable.
foreach(game gol 2048)
  file(GLOB_RECURSE game_sources CONFIGURE_DEPENDS "${ROOT}/demo/${game}/*.cpp")
  list(FILTER game_sources EXCLUDE REGEX "/(main|ai_player|ai_benchmark|game_rules)\\.cpp$|/[^/]*_lib/")
  add_library(ios_${game}_ui STATIC ${game_sources})
  target_link_libraries(ios_${game}_ui PUBLIC sparkium_mobile freetype)
  target_include_directories(ios_${game}_ui PRIVATE "${ROOT}/demo/${game}" "${ROOT}/demo/${game}/${game}_lib" "${ROOT}/demo/gol/game_of_life_lib")
  target_compile_definitions(ios_${game}_ui PRIVATE LONGMARCH_HOSTED_GAMES GLFW_INCLUDE_NONE)
  set(game_generated "${CMAKE_CURRENT_BINARY_DIR}/game_${game}")
  file(MAKE_DIRECTORY "${game_generated}")
  file(GLOB_RECURSE game_shaders CONFIGURE_DEPENDS "${ROOT}/demo/${game}/*.slang")
  add_custom_command(OUTPUT "${game_generated}/built_in_shaders.inl"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.py"
      "${ROOT}/demo/${game}" "${game_generated}/built_in_shaders.inl"
    DEPENDS ${game_shaders} embed_shaders.py VERBATIM)
  target_sources(ios_${game}_ui PRIVATE "${game_generated}/built_in_shaders.inl")
  set_source_files_properties("${ROOT}/demo/${game}/application/application.cpp" PROPERTIES INCLUDE_DIRECTORIES "${game_generated}")
endforeach()
target_sources(sparkium_mobile PRIVATE "${ROOT}/code/grassland/graphics/window.cpp")
