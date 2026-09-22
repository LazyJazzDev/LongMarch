# Sparkium shader sources

- `common/` contains shared shader semantics: BSDFs, material graphs, light
  sampling, geometry helpers, software BVH construction/traversal, and output.
- `raytracing/` contains path-tracing bindings and entry points, including native
  RT hit groups and ray queries.
- `realtime/` contains realtime bindings, visibility, reprojection, tracing,
  filtering and resolve.

`Core::CreatePipelineShadersVFS` combines `common/` and one selected directory.
Paths inside each directory become logical VFS paths, so both renderers can
include `bindings.hlsli` while receiving their own resource layout. Duplicate
logical paths are errors. Generated material headers are added to a copy of that
VFS when compiling scene-specific kernels.

Some common shaders retain `SPARKIUM_SOFTWARE_RT` conditionals for native shader
attributes or callable dispatch. Software tracing defines it before including
these files. This shares semantics without exposing native RT bindings to the
realtime pipeline.

C++ implementations and compiled program caches remain separate. Validate
common shader changes against both renderers, including software and native-query
path-tracing tests. New pipeline-specific entry points belong in their own
folder rather than in `common/`.
