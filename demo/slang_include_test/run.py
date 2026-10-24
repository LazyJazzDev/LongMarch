#!/usr/bin/env python3
"""Exercise Slang's official preprocessing tool without a custom parser."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slangc', default='slangc')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / 'shaders/main.slang'

    def invoke(*options, success=True):
        result = subprocess.run([args.slangc, *map(str, options)], capture_output=True, text=True)
        if success and result.returncode:
            raise RuntimeError(result.stderr)
        if not success and not result.returncode:
            raise RuntimeError('Expected compilation to fail')
        return result

    version = invoke('-version')
    print('Slang:', (version.stdout + version.stderr).strip())
    prefix = ('#define RHI_UNIFORM(T, N) uniform T N;\n'
              '#define RHI_INPUT(T, N, S) T N : S;\n')
    with tempfile.TemporaryDirectory(prefix='slang-include-') as directory:
        root = Path(directory)
        expanded = invoke(source, '-E').stdout
        assert '#include' not in expanded
        assert expanded.count('struct Payload') == 1
        assert expanded.count('float GetValue') == 1
        assert 'RHI_UNIFORM' in expanded and 'RHI_INPUT' in expanded
        assert 'LOCAL_VALUE' not in expanded and '3.0' in expanded
        print('PASS: nested includes, guards and pragma once; undefined RHI macros survive')
        print('Preprocessed source:', expanded.strip())

        # This file is compiled outside the original source tree, without include paths.
        final = root / 'final.slang'
        final.write_text(prefix + expanded, encoding='utf-8')
        for target, suffix in [('spirv', 'spv'), ('metal', 'metal')]:
            output = root / ('final.' + suffix)
            invoke(final, '-entry', 'VSMain', '-stage', 'vertex', '-profile', 'sm_6_0',
                   '-target', target, '-o', output)
            assert output.stat().st_size
            print('PASS: delayed macro definitions compile to', target)

        conditional = root / 'conditional.slang'
        conditional.write_text('#if defined(RHI_INPUT)\nfloat chosen = 1;\n#else\nfloat chosen = 2;\n#endif\n', encoding='utf-8')
        before = invoke(conditional, '-E').stdout
        after = invoke(conditional, '-E', '-DRHI_INPUT=1').stdout
        assert '2' in before and '1' in after and '#if' not in before
        print('PASS: conditional branches are resolved early, not preserved')

        macro_include = root / 'macro_include.slang'
        macro_include.write_text('#define HEADER "types.slang"\n#include HEADER\n', encoding='utf-8')
        assert 'struct Payload' in invoke(macro_include, '-E', '-I', source.parent).stdout
        macro_include.write_text('#include HEADER\n', encoding='utf-8')
        error = invoke(macro_include, '-E', '-I', source.parent)
        assert 'error' in error.stderr
        print('Observation: invalid -E include exit code:', error.returncode)
        print('PASS: macro include paths work only when defined; undefined path fails')
        print('Include diagnostic:', error.stderr.strip())

        arguments = root / 'arguments.slang'
        arguments.write_text('#define FIELD position\nRHI_INPUT(float3, FIELD, POSITION)\n', encoding='utf-8')
        result = invoke(arguments, '-E').stdout
        assert 'RHI_INPUT' in result and 'position' in result and 'FIELD' not in result
        print('PASS: defined macros inside deferred invocations are expanded:', result.strip())

        mapped = invoke(source, '-E', '-line-directive-mode', 'standard').stdout
        print('Observation: -E emits #line:', '#line' in mapped)
        broken = root / 'broken.slang'
        broken.write_text(prefix + expanded.replace('GetValue ( ) ,', 'MissingFunction ( ) ,'), encoding='utf-8')
        diagnostic = invoke(broken, '-entry', 'VSMain', '-stage', 'vertex', '-target', 'spirv', success=False)
        print('Post-preprocessing diagnostic:', diagnostic.stderr.strip())
        assert 'MissingFunction' in diagnostic.stderr
    print('All experiments passed; temporary outputs removed.')


if __name__ == '__main__':
    main()
