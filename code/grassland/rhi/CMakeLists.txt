file(GLOB SOURCES CONFIGURE_DEPENDS "*.h" "*.cpp")

add_library(${GRASSLAND_SUBLIB_NAME} ${SOURCES})

# Cache requests are tied to the RHI shader translation implementation. Both host
# preparation and offline target builds derive the same identity from source.
set(_shader_abi "")
foreach(_shader_source backend/backend_shader_compiler.cpp backend/metal/metal_program_base.cpp rhi_program.h rhi_binding_group.h)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_shader_source}")
    # Git checkout line endings must not change host/target bundle compatibility.
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${_shader_source}" _shader_text)
    string(REPLACE "\r\n" "\n" _shader_text "${_shader_text}")
    string(SHA256 _shader_hash "${_shader_text}")
    string(APPEND _shader_abi "${_shader_hash}")
endforeach()
string(SHA256 _shader_abi "${_shader_abi}")
target_compile_definitions(${GRASSLAND_SUBLIB_NAME} PRIVATE LONGMARCH_SHADER_CACHE_ABI="${_shader_abi}")

list(APPEND GRASSLAND_LIBS ${GRASSLAND_SUBLIB_NAME})
set(GRASSLAND_LIBS ${GRASSLAND_LIBS} PARENT_SCOPE)

target_include_directories(${GRASSLAND_SUBLIB_NAME} PUBLIC ${LONGMARCH_INCLUDE_DIR})

target_link_libraries(${GRASSLAND_SUBLIB_NAME} PRIVATE grassland_util)
if(LONGMARCH_OFFLINE_SHADERS)
    target_compile_definitions(${GRASSLAND_SUBLIB_NAME} PRIVATE LONGMARCH_OFFLINE_SHADERS)
else()
    target_link_libraries(${GRASSLAND_SUBLIB_NAME} PRIVATE slang::slang)
endif()
if(LONGMARCH_HEADLESS)
    target_compile_definitions(${GRASSLAND_SUBLIB_NAME} PRIVATE LONGMARCH_HEADLESS)
else()
    target_link_libraries(${GRASSLAND_SUBLIB_NAME} PUBLIC ${GLFW3_LIB_NAME} grassland_imgui)
endif()

# Display brightness queries are shared by all Windows rendering backends.
if(WIN32)
    target_link_libraries(${GRASSLAND_SUBLIB_NAME} PRIVATE dxgi)
endif()

if (LONGMARCH_CUDA_RUNTIME)
    target_compile_definitions(grassland_rhi PUBLIC LONGMARCH_CUDA_RUNTIME)
    target_link_libraries(grassland_rhi PUBLIC CUDA::cudart_static)
endif ()

add_subdirectory(backend)

# Native interop is opt-in for consumers; do not add its headers to rhi.h.
if(LONGMARCH_METAL_ENABLED)
    target_sources(${GRASSLAND_SUBLIB_NAME} PRIVATE native/native_metal.h native/native_metal.mm)
endif()
