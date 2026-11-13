# RHI shader interfaces

RHI supplies shader declaration macros when creating a program. Shader creation
first resolves source includes and user defines; program creation then expands
the RHI macros and compiles against the selected backend and pipeline layout.

Use `RHI_INPUT(location, type, name)` for vertex-buffer inputs. Its location must
exist in `ProgramDesc::vertex_attributes`.

Use `RHI_VARYING(location, type, name)` for values passed from a vertex shader to
a fragment shader. These locations are independent of vertex-buffer inputs and
need no C++ attribute declaration. Both stages use the same location and
compatible types/interpolation. Field names can differ, but keep declaration
order compatible (including system-value fields): D3D signatures share hardware
registers and require matching register placement, not just matching semantic
numbers. A shared interface struct is the preferred way to keep stages aligned.
The macro unifies declarations; it does not reorder or repack stage signatures.

```slang
struct Surface {
  float4 position : SV_Position;
  RHI_VARYING(0, float2, uv);
  RHI_VARYING(1, float4, color);
  RHI_VARYING(2, nointerpolation uint, material_id);
  RHI_VARYING(3, noperspective float2, screen_offset);
};
```

Ordinary floating-point fields retain perspective-correct interpolation. Put
Slang interpolation modifiers in the type argument, as above; integer fields
should use `nointerpolation`. Location arguments can be numeric macros. Sparse
locations are allowed, subject to backend limits. Matrices/arrays can consume
multiple consecutive locations: reserve the full range and avoid overlaps.

The macro generates the Vulkan location and D3D semantic from the same number;
Metal consumes the SPIR-V interface through the existing SPIRV-Cross path.
Shader authors do not repeat backend-specific location attributes and semantic
indices. `SV_Position`, `SV_TargetN`, `SV_VertexID` and other system semantics
remain explicit: they identify built-in roles, not user varying locations.

`RHI_VARYING` is separate from `RHI_INPUT`: reusing the latter for varyings would
incorrectly require vertex-buffer attributes even for a procedural fullscreen
triangle. Using only automatically assigned stage locations would also couple
separately compiled interfaces to field order and shader permutations.

For interpolation and target mapping, see the [Slang Metal target documentation](https://docs.shader-slang.org/en/stable/external/slang/docs/user-guide/a2-02-metal-target-specific.html),
[D3D shader signature rules](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-signatures)
and [Vulkan location attribute](https://docs.shader-slang.org/en/stable/external/core-module-reference/attributes/vk_location.html).
