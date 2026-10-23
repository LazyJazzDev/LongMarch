file(GLOB SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h")

target_sources(grassland_rhi PRIVATE ${SOURCES})

if (LONGMARCH_VULKAN_ENABLED)
    add_subdirectory(vulkan)
endif ()

if (LONGMARCH_D3D12_ENABLED)
    add_subdirectory(d3d12)
endif ()

if (LONGMARCH_METAL_ENABLED)
    add_subdirectory(metal)
endif ()

# Optional output feedback: X11-only GLFW and older Linux SDKs keep unknown-query defaults.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(RHI_WAYLAND_CLIENT QUIET IMPORTED_TARGET wayland-client)
        pkg_check_modules(RHI_WAYLAND_PROTOCOLS QUIET wayland-protocols>=1.41)
        find_program(RHI_WAYLAND_SCANNER wayland-scanner)
        if(RHI_WAYLAND_CLIENT_FOUND AND RHI_WAYLAND_PROTOCOLS_FOUND AND RHI_WAYLAND_SCANNER)
            include(CheckCXXSourceCompiles)
            include(CMakePushCheckState)
            cmake_push_check_state(RESET)
            set(CMAKE_REQUIRED_LIBRARIES ${GLFW3_LIB_NAME} PkgConfig::RHI_WAYLAND_CLIENT)
            unset(RHI_GLFW_NATIVE_WAYLAND CACHE)
            check_cxx_source_compiles("
                #define GLFW_INCLUDE_NONE
                #define GLFW_EXPOSE_NATIVE_WAYLAND
                #include <GLFW/glfw3.h>
                #include <GLFW/glfw3native.h>
                int main() { return glfwGetPlatform() == GLFW_PLATFORM_WAYLAND && glfwGetWaylandDisplay(); }
            " RHI_GLFW_NATIVE_WAYLAND)
            cmake_pop_check_state()
            if(RHI_GLFW_NATIVE_WAYLAND)
                pkg_get_variable(RHI_WAYLAND_PROTOCOLS_DIR wayland-protocols pkgdatadir)
                set(RHI_COLOR_XML "${RHI_WAYLAND_PROTOCOLS_DIR}/staging/color-management/color-management-v1.xml")
                set(RHI_COLOR_HEADER "${CMAKE_CURRENT_BINARY_DIR}/color-management-v1-client-protocol.h")
                set(RHI_COLOR_CODE "${CMAKE_CURRENT_BINARY_DIR}/color-management-v1-protocol.c")
                add_custom_command(OUTPUT "${RHI_COLOR_HEADER}" "${RHI_COLOR_CODE}"
                    COMMAND "${RHI_WAYLAND_SCANNER}" client-header "${RHI_COLOR_XML}" "${RHI_COLOR_HEADER}"
                    COMMAND "${RHI_WAYLAND_SCANNER}" private-code "${RHI_COLOR_XML}" "${RHI_COLOR_CODE}"
                    DEPENDS "${RHI_COLOR_XML}" VERBATIM)
                add_custom_target(rhi_wayland_color_protocol DEPENDS "${RHI_COLOR_HEADER}" "${RHI_COLOR_CODE}")
                add_dependencies(grassland_rhi rhi_wayland_color_protocol)
                target_sources(grassland_rhi PRIVATE "${RHI_COLOR_HEADER}" "${RHI_COLOR_CODE}")
                target_include_directories(grassland_rhi PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
                target_compile_definitions(grassland_rhi PRIVATE LONGMARCH_RHI_WAYLAND_COLOR)
                target_link_libraries(grassland_rhi PRIVATE PkgConfig::RHI_WAYLAND_CLIENT)
            endif()
        endif()
    endif()
endif()
