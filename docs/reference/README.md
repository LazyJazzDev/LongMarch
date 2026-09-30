# LongMarch architecture and API reference

The reference covers every tracked file below `code/`, including C++ interfaces,
implementations, shaders, binding registration, build definitions and formatting
configuration. Demos and tests are linked as usage and verification evidence.

## Sources of truth

- `modules.json`: architecture, data flow and interface conventions per directory.
- `files.json`: individually reviewed file responsibilities and object/API notes.
- Extracted signatures, declarations, source locations and dependencies come
  directly from the checked-out source. Extraction does not count as review.
- The coverage report distinguishes reviewed documentation from pending files.
  A reference release must have no pending tracked files or missing module notes.

## Build

```sh
python3 -m venv /tmp/longmarch-reference-tools
/tmp/longmarch-reference-tools/bin/pip install -r tools/reference/requirements.txt
/tmp/longmarch-reference-tools/bin/python tools/reference/build.py
```

The generated static pages live in `website/reference/`. Directory navigation
opens separate HTML documents. Source-file pages explain their concrete objects
and interfaces; module pages describe design, ownership and interaction rules.
