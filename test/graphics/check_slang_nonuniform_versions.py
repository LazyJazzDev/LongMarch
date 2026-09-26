"""Run the existing NonUniform corpus plus integer-composition boundary cases.

Use the same --family slang --compiler PATH --output DIR arguments as
compare_nonuniform_compilers.py. Caller-only integer annotations remain diagnostic
cases, not supported shader idioms: annotate at the actual descriptor access.
"""
import compare_nonuniform_compilers as comparison

EXPRESSIONS = {
    "add": "(NonUniformResourceIndex(id.x) + 1) % 2",
    "subtract": "(NonUniformResourceIndex(id.x) - 1) % 2",
    "multiply": "(NonUniformResourceIndex(id.x) * id.x) % 2",
    "divide": "(NonUniformResourceIndex(id.x) / 2) % 2",
    "modulo": "NonUniformResourceIndex(id.x) % 2",
    "shift_left": "(NonUniformResourceIndex(id.x) << 1) % 3",
    "shift_right": "(NonUniformResourceIndex(id.x) >> 1) % 2",
    "bit_and": "NonUniformResourceIndex(id.x) & 1",
    "bit_or": "(NonUniformResourceIndex(id.x) | 2) % 3",
    "bit_xor": "(NonUniformResourceIndex(id.x) ^ 1) % 2",
    "bit_not": "(~NonUniformResourceIndex(id.x)) % 2",
    "negate": "uint(-NonUniformResourceIndex(int(id.x))) % 2",
}

if __name__ == "__main__":
    for name, expression in EXPRESSIONS.items():
        comparison.CASES["arithmetic_" + name] = (
            "ByteAddressBuffer", "",
            f"output[id.x] = buffers[{expression}].Load(0);")
    comparison.main()
