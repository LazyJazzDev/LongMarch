# Optional in-process Clang JIT for the CPU backend's shader graphs.
#
# The CPU backend evaluates shader graphs with an interpreter by default. When
# LLVM and Clang are available it can additionally compile the same generated
# source at runtime, which is what the GPU backends do with DXC.
#
# LLVM is not a vcpkg dependency: the port builds the whole compiler and takes
# hours, and the backend only needs it on desktop development machines. Point
# LLVM_DIR at a system installation instead -- Homebrew's llvm formula, an
# apt llvm-dev package, or the official installer all ship the CMake configs.
#
# LONGMARCH_ENABLE_CPU_JIT: ON builds the JIT if LLVM is found, OFF skips it,
# and AUTO (the default) uses it whenever it is available.
function(LONGMARCH_CONFIGURE_CPU_JIT target)
  set(_mode "${LONGMARCH_ENABLE_CPU_JIT}")
  if (_mode STREQUAL "")
    set(_mode "AUTO")
  endif ()

  if (_mode STREQUAL "OFF")
    message(STATUS "Sparkium CPU JIT: disabled by LONGMARCH_ENABLE_CPU_JIT=OFF")
    return()
  endif ()

  # LLVM_DIR points at the llvm subdirectory of a prefix's lib/cmake, and
  # Clang's config sits beside it.
  find_package(LLVM CONFIG QUIET HINTS
          "$ENV{LLVM_DIR}"
          "/opt/homebrew/opt/llvm/lib/cmake/llvm"
          "/usr/local/opt/llvm/lib/cmake/llvm"
          "/usr/lib/llvm-18/lib/cmake/llvm"
          "/usr/lib/llvm-17/lib/cmake/llvm")
  if (LLVM_FOUND)
    get_filename_component(_llvm_cmake_parent "${LLVM_DIR}" DIRECTORY)
    find_package(Clang CONFIG QUIET HINTS "${_llvm_cmake_parent}/clang" "$ENV{Clang_DIR}")
  endif ()

  if (NOT LLVM_FOUND OR NOT Clang_FOUND)
    if (_mode STREQUAL "ON")
      message(FATAL_ERROR
              "LONGMARCH_ENABLE_CPU_JIT=ON but LLVM/Clang were not found. Set LLVM_DIR to a directory "
              "containing LLVMConfig.cmake, for example $(brew --prefix llvm)/lib/cmake/llvm")
    endif ()
    message(STATUS "Sparkium CPU JIT: LLVM/Clang not found, the interpreter is the only graph engine")
    return()
  endif ()

  target_compile_definitions(${target} PRIVATE LONGMARCH_CPU_JIT_ENABLED=1)
  target_include_directories(${target} SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})

  # clang-cpp carries the compiler; LLVM carries the execution engine. Linking
  # the shared clang-cpp keeps the build fast, and the JIT'd module resolves
  # libc and libc++ symbols from the running process.
  if (TARGET clang-cpp)
    target_link_libraries(${target} PRIVATE clang-cpp LLVM)
  else ()
    target_link_libraries(${target} PRIVATE clangTooling clangFrontend clangDriver clangSerialization
            clangParse clangSema clangEdit clangAST clangLex clangBasic LLVM)
  endif ()
  # No dynamic_lookup or -rdynamic is needed: the JIT'd module only reaches the
  # host through the function pointers in GraphEvalInput, so it never has to
  # resolve a host symbol. Both flags would weaken the executable's own linkage.

  # Clang's own headers (stdarg.h and friends) live beside the libraries, and a
  # driver-less invocation does not find them on its own.
  file(GLOB _clang_resource_dirs "${LLVM_LIBRARY_DIR}/clang/*/include")
  if (_clang_resource_dirs)
    # -resource-dir wants the version directory, which is include's parent.
    list(GET _clang_resource_dirs 0 _clang_resource_include)
    get_filename_component(LONGMARCH_CPU_JIT_RESOURCE_DIR "${_clang_resource_include}" DIRECTORY)
    set(LONGMARCH_CPU_JIT_RESOURCE_DIR "${LONGMARCH_CPU_JIT_RESOURCE_DIR}" PARENT_SCOPE)
  endif ()

  message(STATUS "Sparkium CPU JIT: enabled (LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR})")
endfunction()
