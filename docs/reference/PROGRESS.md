# Reference coverage checkpoint

- Reviewed source inventory: **629 / 629 tracked files under `code/`**.
- Architecture and interface conventions: **75 / 75 directory modules**.
- Object coverage: **5,204 declaration, definition, binding and embedded-source
  records**, including overloads and repeated declarations/implementations.
- All five component families (Contradium, Practium, Grassland, Snowberg and
  Sparkium), Python bindings, shaders and build/configuration files are covered.
- File reviews carry source hashes. Remaining extended-syntax diagnostics have
  explicit hash- and line-checked review records. Python export extraction is
  checked against every registration expression. Embedded shader objects have
  separately reviewed source manifests.
- Module and file navigation opens independent HTML pages. Generated output is
  not committed to the development branch; Pages receives a static snapshot.

`coverage.json` records the exact build checkpoint and remaining issues (none at
this checkpoint). Its source revision precedes documentation-only commits when
no source files changed. Publication regenerates links at the published source
commit. Authoring helpers in temporary directories are not required to rebuild.

The pinned parser versions are tree-sitter 0.25.2 and tree-sitter-cpp 0.23.4;
0.26.0 crashed in the original Python 3.14 environment. See README.md for the
reproducible build and validation commands.
