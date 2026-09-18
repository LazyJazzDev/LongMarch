# Sparkium CPU backend: make the HLSL shader tree compilable as C++.
#
# The shaders are written so that they are also valid C++ once two things are
# handled:
#
#   * `out`/`inout` parameter qualifiers, which have no C++ spelling. They are
#     rewritten to references, so `void f(out float3 x)` becomes
#     `void f(float3& x)`. Both backends then share one source of truth.
#   * `precise`, `register(...)` and the `[numthreads]`/`[shader(...)]`
#     attributes, which the compatibility header and the shader's own
#     SPARKIUM_CPU_SHADER branches deal with.
#
# The rewritten tree is written to the build directory; the checked-in shaders
# are never modified.
function(LONGMARCH_COPY_HLSL_AS_CXX source_dir output_dir)
  file(GLOB_RECURSE shader_files RELATIVE "${source_dir}" "${source_dir}/*.hlsl" "${source_dir}/*.hlsli")
  # The copy happens at configure time, so CMake has to re-run when a shader
  # changes; otherwise a build would silently use the previous rewriting.
  foreach(shader_file ${shader_files})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_dir}/${shader_file}")
  endforeach()
  foreach(shader_file ${shader_files})
    file(READ "${source_dir}/${shader_file}" contents)
    # Anchoring on `(` or `,` keeps the rewrite inside parameter lists, so the
    # prose in the shaders' comments is left alone. The leading group also
    # carries the separator to the replacement, which is what lets several
    # qualified parameters on one line all be rewritten.
    #   `out <type> <name>`   -> `<type>& <name>`   (by reference)
    #   `inout <type> <name>` -> `<type>& <name>`   (by reference)
    #   `in <type> <name>`    -> `<type> <name>`    (by value, HLSL's default)
    string(REGEX REPLACE
      "(^|[(,][ \t\r\n]*)(inout|out)[ \t\r\n]+([A-Za-z_][A-Za-z0-9_]*)[ \t\r\n]+([A-Za-z_][A-Za-z0-9_]*)"
      "\\1\\3& \\4" contents "${contents}")
    string(REGEX REPLACE
      "(^|[(,][ \t\r\n]*)(in)[ \t\r\n]+([A-Za-z_][A-Za-z0-9_]*)[ \t\r\n]+([A-Za-z_][A-Za-z0-9_]*)"
      "\\1\\3 \\4" contents "${contents}")
    # HLSL gives a `class`'s members public access by default, the same as
    # `struct`; C++ defaults them to private. Emitting the definitions as
    # structs keeps the shaders' cross-object accesses compiling. Every `class`
    # at the start of a line in this tree introduces a type, and template
    # parameter lists never begin a line with one.
    string(REGEX REPLACE "\n(class)[ \t]+([A-Za-z_][A-Za-z0-9_]*)" "\nstruct \\2" contents "${contents}")
    # `precise` asks for IEEE-strict evaluation. Float arithmetic on the CPU
    # targets already is IEEE-strict, so the qualifier only has to disappear.
    # CMake's regex has no \b, hence the explicit "not part of an identifier"
    # group, which also keeps the leading character.
    string(REGEX REPLACE "(^|[^A-Za-z0-9_])precise[ \t\r\n]+" "\\1" contents "${contents}")
    # HLSL zero-initialises a plain struct through a C-style cast, which C++
    # has no conversion for; aggregate initialisation says the same thing.
    string(REGEX REPLACE "\\((SoftwareHit)\\)0" "\\1{}" contents "${contents}")
    get_filename_component(shader_dir "${output_dir}/${shader_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${shader_dir}")
    file(WRITE "${output_dir}/${shader_file}" "${contents}")
  endforeach()
endfunction()
