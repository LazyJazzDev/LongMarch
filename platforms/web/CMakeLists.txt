cmake_minimum_required(VERSION 3.25)
# Browser apps (2048, Game of Life, N-body) on WebGPU. Configure with emcmake;
# see README.md. Shaders come precompiled to WGSL (prepare_shaders.cpp).
project(LongMarchWeb LANGUAGES C CXX)
if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Configure the web apps with emcmake cmake")
endif()
set(CMAKE_CXX_STANDARD 17)
set(WEB_RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../out/web/resources" CACHE PATH
  "Prepared resources: shaders/ from web_shader_prepare and assets/fonts")
if(NOT EXISTS "${WEB_RESOURCES}/shaders")
  message(FATAL_ERROR "Prepare WEB_RESOURCES first; see README.md")
endif()
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(MOBILE "${ROOT}/platforms/ios")
# The engine code throws; WebAssembly exceptions are native in current browsers.
# FreeType (added by the mobile dependencies) uses setjmp/longjmp, which pairs with the same exception model.
add_compile_options(-fwasm-exceptions -sSUPPORT_LONGJMP=wasm)
add_link_options(-fwasm-exceptions -sSUPPORT_LONGJMP=wasm)
include("${ROOT}/platforms/mobile/Dependencies.cmake")
set(HEADERS "${LONGMARCH_MOBILE_HEADERS}")
find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(GLOB UTIL "${ROOT}/code/grassland/util/*.cpp")
file(GLOB GRAPHICS "${ROOT}/code/grassland/graphics/*.cpp")
file(GLOB WEBGPU "${ROOT}/code/grassland/graphics/backend/webgpu/*.cpp")
add_library(web_engine STATIC ${UTIL} ${GRAPHICS} ${WEBGPU}
  "${ROOT}/code/grassland/math/math_mesh.cxx" "${mikktspace_SOURCE_DIR}/mikktspace.c"
  "${ROOT}/demo/gol/game_of_life_lib/game_of_life_lib.cpp"
  "${ROOT}/demo/2048/2048_lib/2048_lib.cpp" "${ROOT}/demo/2048/game_rules.cpp" "${ROOT}/demo/2048/ai_player.cpp")
target_include_directories(web_engine PUBLIC "${ROOT}/code" "${HEADERS}" "${HEADERS}/eigen3" "${HEADERS}/stb" "${ROOT}"
  "${ROOT}/demo/2048/2048_lib" "${mikktspace_SOURCE_DIR}")
target_compile_definitions(web_engine PUBLIC LONGMARCH_WEBGPU_ENABLED LONGMARCH_HEADLESS LONGMARCH_OFFLINE_SHADERS
  FMT_HEADER_ONLY NDEBUG GLFW_INCLUDE_NONE "LONGMARCH_ASSETS_DIR=\"\"")
target_compile_options(web_engine PUBLIC --use-port=emdawnwebgpu)
target_link_options(web_engine PUBLIC --use-port=emdawnwebgpu)

function(embed_shaders root output)
  file(GLOB_RECURSE shaders CONFIGURE_DEPENDS "${root}/*.slang")
  add_custom_command(OUTPUT "${output}"
    COMMAND "${Python3_EXECUTABLE}" "${MOBILE}/embed_shaders.py" "${root}" "${output}"
    DEPENDS ${shaders} "${MOBILE}/embed_shaders.py" VERBATIM)
endfunction()

# grassland/graphics/shader.cpp embeds the engine's own Slang library.
set(engine_shaders "${CMAKE_CURRENT_BINARY_DIR}/grassland_graphics")
file(MAKE_DIRECTORY "${engine_shaders}")
embed_shaders("${ROOT}/code/grassland/graphics" "${engine_shaders}/built_in_shaders.inl")
target_sources(web_engine PRIVATE "${engine_shaders}/built_in_shaders.inl")
set_source_files_properties("${ROOT}/code/grassland/graphics/shader.cpp" PROPERTIES INCLUDE_DIRECTORIES "${engine_shaders}")

# The desktop game UIs, in the same namespaces as the mobile apps.
foreach(game gol 2048)
  file(GLOB_RECURSE game_sources CONFIGURE_DEPENDS "${ROOT}/demo/${game}/*.cpp")
  list(FILTER game_sources EXCLUDE REGEX "/(main|ai_player|ai_benchmark|game_rules)\\.cpp$|/[^/]*_lib/")
  set(generated "${CMAKE_CURRENT_BINARY_DIR}/game_${game}")
  file(MAKE_DIRECTORY "${generated}")
  embed_shaders("${ROOT}/demo/${game}" "${generated}/built_in_shaders.inl")
  add_library(web_${game} STATIC ${game_sources} "${generated}/built_in_shaders.inl")
  target_link_libraries(web_${game} PUBLIC web_engine freetype)
  target_include_directories(web_${game} PRIVATE "${ROOT}/demo/${game}" "${ROOT}/demo/${game}/${game}_lib"
    "${ROOT}/demo/gol/game_of_life_lib")
  target_compile_definitions(web_${game} PRIVATE LONGMARCH_HOSTED_GAMES)
  set_source_files_properties("${ROOT}/demo/${game}/application/application.cpp" PROPERTIES
    INCLUDE_DIRECTORIES "${generated}")
endforeach()

embed_shaders("${ROOT}/demo" "${CMAKE_CURRENT_BINARY_DIR}/demo_shaders.inl")
add_executable(longmarch main.cpp "${MOBILE}/demos/DemoSession.cpp" "${MOBILE}/games/DesktopGameSession.cpp"
  "${MOBILE}/games/image_write.cpp" "${CMAKE_CURRENT_BINARY_DIR}/demo_shaders.inl")
target_include_directories(longmarch PRIVATE "${MOBILE}" "${CMAKE_CURRENT_BINARY_DIR}" "${ROOT}/demo/gol/game_of_life_lib"
  "${ROOT}/demo/2048" "${ROOT}/demo/gol")
target_link_libraries(longmarch PRIVATE web_gol web_2048)
target_link_options(longmarch PRIVATE -sENVIRONMENT=web -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=0
  "--preload-file=${WEB_RESOURCES}@/res")
set_target_properties(longmarch PROPERTIES LINK_DEPENDS "${WEB_RESOURCES}/shaders")
# The page (index.html) loads longmarch.js next to it.
add_custom_command(TARGET longmarch POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/index.html" "$<TARGET_FILE_DIR:longmarch>/index.html"
  VERBATIM)
