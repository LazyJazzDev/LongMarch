"""Compile the complete storage-buffer NonUniform comparison corpus.

Requires spirv-dis and spirv-val on PATH. Outputs raw modules, diagnostics,
source, disassembly and results.json. Does not apply LongMarch repairs.
The pointer inspection is specific to this corpus, not general SPIR-V analysis.
"""

import argparse
import json
from pathlib import Path
import struct
import subprocess

CASES = {'byte_direct': ('ByteAddressBuffer',
                 '',
                 'output[id.x] = buffers[NonUniformResourceIndex(id.x%2)].Load(0);'),
 'byte_local': ('ByteAddressBuffer',
                '',
                'ByteAddressBuffer b = buffers[NonUniformResourceIndex(id.x%2)]; output[id.x] = '
                'b.Load(0);'),
 'byte_index_local': ('ByteAddressBuffer',
                      '',
                      'uint i=NonUniformResourceIndex(id.x%2); output[id.x] = buffers[i].Load(0);'),
 'byte_param': ('ByteAddressBuffer',
                'uint Read(ByteAddressBuffer b) { return b.Load(0); }',
                'output[id.x] = Read(buffers[NonUniformResourceIndex(id.x%2)]);'),
 'byte_return': ('ByteAddressBuffer',
                 'ByteAddressBuffer Pick(uint i) { return buffers[NonUniformResourceIndex(i)]; }',
                 'output[id.x] = Pick(id.x%2).Load(0);'),
 'byte_helper_index': ('ByteAddressBuffer',
                       'uint Read(uint i) { return buffers[NonUniformResourceIndex(i)].Load(0); }',
                       'output[id.x] = Read(id.x%2);'),
 'byte_templated_local': ('ByteAddressBuffer',
                          '',
                          'ByteAddressBuffer b = buffers[NonUniformResourceIndex(id.x%2)]; '
                          'output[id.x] = b.Load<uint>(0);'),
 'structured_direct': ('StructuredBuffer<uint>',
                       '',
                       'output[id.x] = buffers[NonUniformResourceIndex(id.x%2)][0];'),
 'structured_local': ('StructuredBuffer<uint>',
                      '',
                      'StructuredBuffer<uint> b = buffers[NonUniformResourceIndex(id.x%2)]; '
                      'output[id.x] = b[0];'),
 'rwbyte_local': ('RWByteAddressBuffer',
                  '',
                  'RWByteAddressBuffer b = buffers[NonUniformResourceIndex(id.x%2)]; output[id.x] '
                  '= b.Load(0);'),
 'byte_helper_annotated_arg': ('ByteAddressBuffer',
                               'uint Read(uint i) { return buffers[i].Load(0); }',
                               'output[id.x] = Read(NonUniformResourceIndex(id.x%2));'),
 'byte_index_rewrapped': ('ByteAddressBuffer',
                          '',
                          'uint i=NonUniformResourceIndex(id.x%2); output[id.x] = '
                          'buffers[NonUniformResourceIndex(i)].Load(0);'),
 'fixed_byte_local': ('ByteAddressBuffer',
                      '',
                      'ByteAddressBuffer b = buffers[NonUniformResourceIndex(id.x%2)]; '
                      'output[id.x] = b.Load(0);'),
 'fixed_byte_direct': ('ByteAddressBuffer',
                       '',
                       'output[id.x] = buffers[NonUniformResourceIndex(id.x%2)].Load(0);'),
 'byte_uniform': ('ByteAddressBuffer',
                  '',
                  'ByteAddressBuffer b = buffers[0]; output[id.x] = b.Load(0);'),
 'byte_noinline_arg': ('ByteAddressBuffer',
                       '[noinline] uint Read(uint i) { return buffers[i].Load(0); }',
                       'output[id.x] = Read(NonUniformResourceIndex(id.x%2));'),
 'byte_noinline_inside': ('ByteAddressBuffer',
                          '[noinline] uint Read(uint i) { return '
                          'buffers[NonUniformResourceIndex(i)].Load(0); }',
                          'output[id.x] = Read(id.x%2);')}


def inspect(data):
    words = struct.unpack(f"<{len(data) // 4}I", data)
    instructions = []
    offset = 5
    while offset < len(words):
        count = words[offset] >> 16
        if not count or offset + count > len(words):
            raise ValueError("Malformed SPIR-V instruction")
        instructions.append((words[offset] & 65535, words[offset + 1:offset + count]))
        offset += count
    names = {
        args[0]: struct.pack(f"<{len(args) - 1}I", *args[1:]).split(b"\0")[0].decode()
        for opcode, args in instructions if opcode == 5
    }
    roots = {identifier for identifier, name in names.items() if name == "buffers"}
    chains = {args[1]: args[2] for opcode, args in instructions if opcode in (65, 66)}
    decorations = {args[0] for opcode, args in instructions
                   if opcode == 71 and args[1] == 5300}
    while True:
        expanded = roots | {identifier for identifier, base in chains.items() if base in roots}
        if expanded == roots:
            break
        roots = expanded
    pointers = [args[2] for opcode, args in instructions if opcode == 61 and args[2] in roots]
    return {
        "load_pointers": len(pointers),
        "decorated": sum(pointer in decorations for pointer in pointers),
        "capability": any(opcode == 17 and args[0] == 5308 for opcode, args in instructions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("dxc", "slang"), required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    version_flag = "--version" if args.family == "dxc" else "-version"
    version_result = subprocess.run([args.compiler, version_flag], capture_output=True,
                                    text=True, check=True)
    version = (version_result.stdout + version_result.stderr).strip()
    (args.output / "version.txt").write_text(version + "\n")
    rows = []
    for name, (resource_type, helper, body) in CASES.items():
        for optimization in (["-Od", "-O3"] if args.family == "dxc" else ["-O0", "-O3"]):
            stem = args.output / (name + optimization)
            source, module = stem.with_suffix(".hlsl"), stem.with_suffix(".spv")
            size = "2" if name.startswith("fixed_") else ""
            register = "u" if resource_type.startswith("RW") else "t"
            source.write_text(
                f"{resource_type} buffers[{size}] : register({register}0, space0);\n"
                "RWStructuredBuffer<uint> output : register(u0, space1);\n"
                f"{helper}\n[numthreads(64, 1, 1)]\n"
                f"void Main(uint3 id : SV_DispatchThreadID) {{ {body} }}\n")
            if args.family == "dxc":
                command = [args.compiler, "-spirv", "-T", "cs_6_0", "-E", "Main",
                           "-fspv-target-env=vulkan1.2", optimization, str(source), "-Fo", str(module)]
            else:
                command = [args.compiler, str(source), "-entry", "Main", "-stage", "compute",
                           "-target", "spirv", "-profile", "spirv_1_5", "-emit-spirv-directly",
                           "-fvk-t-shift", "0", "all", "-fvk-u-shift", "0", "all",
                           optimization, "-o", str(module)]
            result = subprocess.run(command, capture_output=True, text=True)
            stem.with_suffix(".log").write_text(result.stdout + result.stderr)
            row = dict(case=name, opt=optimization, command=command, compile_exit=result.returncode)
            if result.returncode:
                row["status"] = "compile_error"
            else:
                row.update(inspect(module.read_bytes()))
                validation = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(module)],
                                            capture_output=True, text=True)
                row["spirv_val"] = validation.returncode
                stem.with_suffix(".val.log").write_text(validation.stdout + validation.stderr)
                assembly = subprocess.check_output(["spirv-dis", str(module)], text=True)
                stem.with_suffix(".spvasm").write_text(assembly)
                # Zero/multiple pointers require manual analysis instead of a false success.
                if validation.returncode or row["load_pointers"] != 1:
                    row["status"] = "needs_review"
                elif name == "byte_uniform":
                    row["status"] = "uniform_control"
                elif row["decorated"] == 1 and row["capability"]:
                    row["status"] = "correct"
                elif row["decorated"] == 1:
                    row["status"] = "missing_capability"
                elif row["capability"]:
                    row["status"] = "missing_decoration"
                else:
                    row["status"] = "missing_both"
            rows.append(row)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(version)
    for status in sorted({row["status"] for row in rows}):
        print(status, sum(row["status"] == status for row in rows))


if __name__ == "__main__":
    main()
