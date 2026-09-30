# LongMarch architecture and API reference

The Chinese reference covers every tracked file under `code/`: C++/CUDA
interfaces and implementations, shaders, Python bindings, build definitions and
configuration. Demos and tests are linked as usage and verification evidence.

## Sources of truth

- `modules.json`: architecture, data flow and interface conventions for each directory.
- `files.json`: individually authored responsibilities and implementation notes,
  pinned to each reviewed file's SHA-256.
- `api.json`: semantic descriptions and caller contracts for concrete objects.
  File-qualified keys disambiguate identically named helpers and shader entries.
- `supplemental.json`: reviewed embedded shader objects and raw-string inventory.
- `diagnostics.json`: explicit, versioned review of remaining parser diagnostics
  caused by CUDA, Objective-C, conditional compilation and C++ extensions.
- `coverage.json`: generated coverage checkpoint. Declaration and definition
  records include overloads and repeated declarations; they are not unique APIs.

Signatures, source locations, parameters, fields and implementation expressions
come directly from the checked-out source. Extraction does not count as review.
Python registration expressions are indexed alongside their C++ registrars.
Private/internal objects are identified by their source access level; their
presence in this reference does not make them supported public APIs.

## Build and validate

```sh
python3 -m venv /tmp/longmarch-reference-tools
/tmp/longmarch-reference-tools/bin/pip install -r tools/reference/requirements.txt
/tmp/longmarch-reference-tools/bin/python tools/reference/build.py --strict
/tmp/longmarch-reference-tools/bin/python tools/reference/validate.py
```

Strict builds reject missing/stale file reviews, missing object descriptions,
missing module designs, unreviewed parser diagnostics, changed embedded-source
inventories and incomplete Python registration extraction. The HTML validator
checks internal links, fragment targets, duplicate IDs and independent sidebar
navigation across the generated site.

Generated pages live in ignored `website/reference/`. Each module and source
file has its own HTML document. The hierarchical sidebar filters all modules
and files. Source links pin the Git revision used for that build. Pages publishes
a generated snapshot from `gh-pages`; after the development PR is merged, the
main workflow can build the same reference directly from its versioned sources.

Coverage checks detect documentation drift; they do not prove runtime correctness
or replace review of the prose when source behavior changes.
