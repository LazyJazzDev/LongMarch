file(GLOB SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")
add_library(${SPARKIUM_RAYTRACING_SUBLIB_NAME} ${SOURCES})

# The CPU backend compiles the shared HLSL sources as C++. They are copied into
# the build directory with the HLSL-only syntax rewritten there; see
# cmake/HlslAsCxx.cmake. The checked-in shaders are never modified.
include(${CMAKE_SOURCE_DIR}/cmake/HlslAsCxx.cmake)
LONGMARCH_COPY_HLSL_AS_CXX(${LONGMARCH_SPARKIUM_SHADERS} "${CMAKE_CURRENT_BINARY_DIR}/shaders")

# The rewritten tree resolves the shaders' relative includes; the shim directory
# holds the HLSL compatibility layer the sources are compiled against.
# Public so tests can exercise the compatibility layer and the shipped shaders
# directly, which is how the traversal is checked against an oracle. The names
# these headers add live in namespaces, not at global scope.
target_include_directories(${SPARKIUM_RAYTRACING_SUBLIB_NAME} PUBLIC
        "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders")

# float3 reaches x/y/z through an anonymous struct in a union, which is how HLSL
# vectors are laid out and is accepted by every compiler this project targets.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${SPARKIUM_RAYTRACING_SUBLIB_NAME} PRIVATE -Wno-gnu-anonymous-struct)
endif ()

# The shaders and the BVH builder need nothing from the rest of sparkium,
# which lets the core sublibrary link this one instead of the other way round.
# The shader-graph JIT, when LLVM and Clang are available. The include list it
# is given is what the runtime compilation resolves the compatibility layer and
# the rewritten shader tree through.
include(${CMAKE_SOURCE_DIR}/cmake/CpuJit.cmake)
LONGMARCH_CONFIGURE_CPU_JIT(${SPARKIUM_RAYTRACING_SUBLIB_NAME})
# The macOS SDK has to be given to the driver explicitly: an out-of-tree Clang
# has no default sysroot, so it would not find the C++ standard library.
if (LONGMARCH_CPU_JIT_RESOURCE_DIR)
    target_compile_definitions(${SPARKIUM_RAYTRACING_SUBLIB_NAME} PRIVATE
            "LONGMARCH_CPU_JIT_RESOURCE_DIR=\"${LONGMARCH_CPU_JIT_RESOURCE_DIR}\"")
endif ()
if (APPLE)
    execute_process(COMMAND xcrun --show-sdk-path
            OUTPUT_VARIABLE LONGMARCH_CPU_JIT_SDK OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if (LONGMARCH_CPU_JIT_SDK)
        target_compile_definitions(${SPARKIUM_RAYTRACING_SUBLIB_NAME} PRIVATE
                "LONGMARCH_CPU_JIT_SYSROOT=\"${LONGMARCH_CPU_JIT_SDK}\"")
    endif ()
endif ()

# Comma separated rather than a CMake list: the definition is passed straight to
# the compiler, and a semicolon or an embedded quote does not survive that.
set(LONGMARCH_CPU_JIT_INCLUDE_DIRS
        "${CMAKE_SOURCE_DIR}/code,${CMAKE_CURRENT_BINARY_DIR}/shaders,${CMAKE_CURRENT_SOURCE_DIR}/shaders")
target_compile_definitions(${SPARKIUM_RAYTRACING_SUBLIB_NAME} PRIVATE
        "LONGMARCH_CPU_JIT_INCLUDE_DIRS=\"${LONGMARCH_CPU_JIT_INCLUDE_DIRS}\"")

target_link_libraries(${SPARKIUM_RAYTRACING_SUBLIB_NAME} PUBLIC Grassland Snowberg)
list(APPEND SPARKIUM_RAYTRACING_LIBS ${SPARKIUM_RAYTRACING_SUBLIB_NAME})
set(SPARKIUM_RAYTRACING_LIBS ${SPARKIUM_RAYTRACING_LIBS} PARENT_SCOPE)
