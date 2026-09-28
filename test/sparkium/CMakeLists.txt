add_executable(sparkium_fallback_test fallback_test.cpp metal_backend_test.cpp hdr_window_test.cpp)
if (LONGMARCH_METAL_ENABLED)
    target_sources(sparkium_fallback_test PRIVATE metal_window_test.mm)
endif ()
target_link_libraries(sparkium_fallback_test PRIVATE LongMarch GTest::gtest_main)
LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(sparkium_fallback_test)
add_test(NAME sparkium_fallback COMMAND sparkium_fallback_test)

# Optional diagnostic only: no Wayland protocol/header dependency for normal
# builds, including Linux/X11, Windows and macOS.
option(LONGMARCH_ENABLE_WAYLAND_COLOR_PROBE "Build the native Wayland reference-white diagnostic" OFF)
if (LONGMARCH_ENABLE_WAYLAND_COLOR_PROBE)
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT LONGMARCH_VULKAN_ENABLED)
        message(FATAL_ERROR "The Wayland color probe requires Linux and Vulkan")
    endif ()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(WAYLAND_CLIENT REQUIRED IMPORTED_TARGET wayland-client)
    pkg_check_modules(WAYLAND_PROTOCOLS REQUIRED wayland-protocols>=1.41)
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES glfw PkgConfig::WAYLAND_CLIENT)
    unset(LONGMARCH_GLFW_NATIVE_WAYLAND CACHE)
    check_cxx_source_compiles("
        #define GLFW_EXPOSE_NATIVE_WAYLAND
        #include <GLFW/glfw3.h>
        #include <GLFW/glfw3native.h>
        int main() { return glfwGetWaylandDisplay() == nullptr; }
    " LONGMARCH_GLFW_NATIVE_WAYLAND)
    cmake_pop_check_state()
    if (NOT LONGMARCH_GLFW_NATIVE_WAYLAND)
        message(FATAL_ERROR "The color probe requires GLFW native Wayland support; enable the wayland manifest feature")
    endif ()
    pkg_get_variable(WAYLAND_PROTOCOLS_DIR wayland-protocols pkgdatadir)
    find_program(WAYLAND_SCANNER wayland-scanner REQUIRED)
    set(COLOR_PROTOCOL_XML "${WAYLAND_PROTOCOLS_DIR}/staging/color-management/color-management-v1.xml")
    set(COLOR_PROTOCOL_HEADER "${CMAKE_CURRENT_BINARY_DIR}/color-management-v1-client-protocol.h")
    set(COLOR_PROTOCOL_CODE "${CMAKE_CURRENT_BINARY_DIR}/color-management-v1-protocol.c")
    add_custom_command(OUTPUT "${COLOR_PROTOCOL_HEADER}" "${COLOR_PROTOCOL_CODE}"
        COMMAND "${WAYLAND_SCANNER}" client-header "${COLOR_PROTOCOL_XML}" "${COLOR_PROTOCOL_HEADER}"
        COMMAND "${WAYLAND_SCANNER}" private-code "${COLOR_PROTOCOL_XML}" "${COLOR_PROTOCOL_CODE}"
        DEPENDS "${COLOR_PROTOCOL_XML}" VERBATIM)
    add_executable(wayland_color_probe wayland_color_probe.cpp "${COLOR_PROTOCOL_CODE}" "${COLOR_PROTOCOL_HEADER}")
    target_include_directories(wayland_color_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(wayland_color_probe PRIVATE LongMarch PkgConfig::WAYLAND_CLIENT)
    LONGMARCH_DEPLOY_RUNTIME_DEPENDENCIES(wayland_color_probe)
endif ()
