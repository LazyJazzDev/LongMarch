# RHI shader interfaces

Use `RHI_INPUT(location, type, name)` for vertex-buffer inputs. Its location must
exist in `ProgramDesc::vertex_attributes`. Resource declarations likewise use
the RHI binding macros to follow the C++ pipeline layout.

For vertex-to-fragment values, use ordinary Slang semantics with matching types,
interpolation modifiers and declaration order. Prefer a shared interface struct:

```slang
struct Surface {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
  float4 color : TEXCOORD1;
  nointerpolation uint material_id : TEXCOORD2;
  noperspective float2 screen_offset : TEXCOORD3;
};
```

Slang assigns Vulkan locations automatically. The numeric suffix of `TEXCOORDN`
is a semantic index, not an explicit Vulkan location. Do not reorder fields or
omit fields from the middle of the fragment interface while assuming that
matching semantic names alone will preserve the Vulkan mapping. D3D also
requires compatible signature register placement. Field names can differ.

A fragment interface can consume a matching prefix of the vertex output; this
is how Sparkium's Lambertian, light and specular materials omit the trailing
tangent fields used by the principled material. Keep that prefix in the same
order with the same types and semantics. Conditional interfaces must preserve
this agreement across both shader compilations.

Ordinary floating-point fields retain perspective-correct interpolation;
`nointerpolation` and `noperspective` retain their normal Slang meanings. Metal
uses the generated SPIR-V interface through the existing SPIRV-Cross path.
System semantics such as `SV_Position`, `SV_TargetN` and `SV_VertexID` remain
explicit. No separate stage-interface macro or handwritten Vulkan location
annotation is needed for the repository's current interfaces.

See [Slang's cross-stage matching rules](https://github.com/shader-slang/slang/blob/master/docs/user-guide/02-conventional-features.md)
and [D3D shader signature rules](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-signatures).
