"""Extract source evidence, preserving definitions even when C++ extensions fail.

This module does not generate semantic API descriptions. Those are maintained in
reviewed docs/reference records; source extraction supplies verifiable evidence.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_cpp

PARSER = Parser(Language(tree_sitter_cpp.language()))
CPP_SUFFIXES = {'.h', '.cpp', '.cuh', '.cu', '.cxx', '.mm', '.slang', '.vert', '.frag'}


def tracked_files(root: Path) -> list[Path]:
    paths = subprocess.check_output(['git', 'ls-files', '-z', 'code'], cwd=root).decode().split('\0')
    return [root / path for path in sorted(paths) if path and (root / path).is_file()]


def text(node, data):
    return data[node.start_byte:node.end_byte].decode('utf-8', errors='replace') if node else ''


def walk(node):
    yield node
    for child in node.named_children:
        yield from walk(child)


def declarator_name(node, data):
    if node is None:
        return ''
    while node.child_by_field_name('declarator') is not None:
        node = node.child_by_field_name('declarator')
    return text(node, data)


def preceding_comment(node, data):
    comments = []
    prev = node.prev_named_sibling
    while prev and prev.type == 'comment' and node.start_point.row - prev.end_point.row <= 2:
        comments.insert(0, text(prev, data))
        node, prev = prev, prev.prev_named_sibling
    return '\n'.join(comments)


def extract(path: Path, root: Path) -> dict:
    data = path.read_bytes()
    source = data.decode('utf-8', errors='replace')
    record = {
        'path': path.relative_to(root).as_posix(),
        'sha256': hashlib.sha256(data).hexdigest(),
        'lines': len(source.splitlines()),
        'includes': re.findall(r'^\s*#\s*include\s*[<"]([^>"]+)', source, re.M),
        'objects': [],
        'parse_errors': [],
        'source': source,
    }
    if path.suffix not in CPP_SUFFIXES:
        return record
    # Ignore calling-convention annotations for parsing, retaining exact offsets
    # and the unmodified source for displayed signatures and line evidence.
    parse_data = re.sub(rb"(?m)^(?![ \t]*#)([^\n]*)$",
                        lambda m: re.sub(rb"\b(?:LM_DEVICE_FUNC|__host__|__device__|__global__|__forceinline__|VKAPI_ATTR|VKAPI_CALL)\b",
                                         lambda token: b" " * len(token.group()), m.group()), data)
    prefixes = []
    # Shader storage blocks and GLSL qualifiers need a structural C++ surrogate.
    if path.suffix in {'.slang', '.vert', '.frag'}:
        parse_data = re.sub(rb'\bcbuffer\b', lambda m: b'struct ', parse_data)
        parse_data = re.sub(rb'\bprecise\b|\bgroupshared\b', lambda m: b' ' * len(m.group()), parse_data)
    if path.suffix in {'.vert', '.frag'}:
        parse_data = re.sub(rb'layout\([^)]*\)', lambda m: b' ' * len(m.group()), parse_data)
        parse_data = re.sub(rb'\bbuffer(?=\s+\w+\s*\{)', lambda m: b'struct', parse_data)
        parse_data = re.sub(rb'\b(?:in|out|uniform)\b', lambda m: b' ' * len(m.group()), parse_data)
    if path.suffix == '.slang':
        # Adapt declaration-only Slang syntax, keeping byte offsets and original
        # text. Without this, C++ recovery mislabels function locals as globals.
        def blank_prefix(match):
            prefixes.append((match.start(), match.end()))
            return re.sub(rb'[^\n]', b' ', match.group())
        parse_data = re.sub(rb'__generic\s*<[^>]+>|\[(?:mutating|unroll|maxvertexcount\([^\]]*\)|shader\([^\]]*\)|numthreads\([^\]]*\))\]',
                            blank_prefix, parse_data)
        parse_data = re.sub(rb'\binterface\b',
                            lambda m: b'struct' + b' ' * (len(m.group()) - 6), parse_data)
        parse_data = re.sub(rb'\bextension(\s+\w+)\s*:\s*\w+',
                            lambda m: b'namespace' + m.group(1) + b' ' * (len(m.group()) - 9 - len(m.group(1))), parse_data)
        parse_data = re.sub(rb'\b(?:inout|out|in|triangle|SP_MUTATING)\b',
                            lambda m: b' ' * len(m.group()), parse_data)
        # Slang register bindings and semantics are not C++ base classes or
        # function declarators. Blank only their annotation bytes so original
        # signatures and source locations remain intact.
        parse_data = re.sub(rb':\s*register\s*\([^)]*\)',
                            lambda m: re.sub(rb'[^\n]', b' ', m.group()), parse_data)
        parse_data = re.sub(rb':\s*(?:SV_[A-Za-z0-9_]+|TEXCOORD[0-9]*|POSITION[0-9]*|COLOR[0-9]*)(?![A-Za-z0-9_])',
                            lambda m: re.sub(rb'[^\n]', b' ', m.group()), parse_data)
    parse_data = re.sub(rb'=\s*\{[^{}]*\}', lambda m: b'=' + re.sub(rb'[^\n]', b' ', m.group()[1:-1]) + b'0', parse_data)
    tree = PARSER.parse(parse_data)
    record['parse_errors'] = sorted(set(n.start_point.row + 1 for n in walk(tree.root_node) if n.type == 'ERROR' or n.is_missing))

    def visit(node, scope='', access='public', conditions=()):
        kind = node.type
        if kind == 'namespace_definition':
            name = text(node.child_by_field_name('name'), data) or '(anonymous)'
            if text(node, data).startswith('extension'):
                body = node.child_by_field_name('body')
                record['objects'].append({
                    'name': name, 'qualified': scope + name, 'kind': 'slang_extension',
                    'line': node.start_point.row + 1, 'end_line': node.end_point.row + 1,
                    'signature': data[node.start_byte:body.start_byte].decode().strip(),
                    'access': access, 'conditions': list(conditions),
                    'comment': preceding_comment(node, data), 'members': [], 'parameters': [],
                    'calls': [], 'returns': [], 'throws': [],
                })
            visit(node.child_by_field_name('body'), f'{scope}{name}::', 'public', conditions)
            return
        if kind.startswith('preproc_if') or kind == 'preproc_else':
            condition = text(node.child_by_field_name('condition') or node.child_by_field_name('name'), data)
            if not condition:
                condition = text(node, data).splitlines()[0]
            conditions = (*conditions, condition)
        if kind == 'friend_declaration':
            parent_scope = scope.rsplit('::', 2)[0] + '::' if '::' in scope.rstrip(':') else ''
            for child in node.named_children:
                visit(child, parent_scope, access, conditions)
            return
        is_type = kind in {'class_specifier', 'struct_specifier', 'enum_specifier', 'union_specifier'}
        function = None
        if kind in {'function_definition', 'declaration', 'field_declaration'}:
            declarator = node.child_by_field_name('declarator')
            if declarator:
                function = next((n for n in walk(declarator) if n.type in {'function_declarator', 'operator_cast'}), None)
            # Parameters are not top-level callable declarations.
        is_variable = kind == 'declaration' and not function and node.child_by_field_name('declarator') is not None
        if is_type or function or is_variable or kind in {'alias_declaration', 'using_declaration', 'type_definition', 'preproc_def', 'preproc_function_def'}:
            name = (text(node.child_by_field_name('name'), data) if is_type else
                    declarator_name(function.child_by_field_name('declarator'), data) if function else
                    text(node.child_by_field_name('name'), data) or declarator_name(node.child_by_field_name('declarator'), data))
            if function and function.type == 'operator_cast':
                cast = text(node.child_by_field_name('declarator'), data)
                name = cast.split('(')[0].strip()
            if kind == 'using_declaration':
                name = re.sub(r'^using\s+(?:namespace\s+)?|;$', '', text(node, data)).strip()
            name = name or '(anonymous)'
            body = node.child_by_field_name('body')
            signature_start = node.start_byte
            if node.parent and node.parent.type == 'template_declaration':
                signature_start = node.parent.start_byte
            for prefix_start, prefix_end in reversed(prefixes):
                if prefix_end <= signature_start and not parse_data[prefix_end:signature_start].strip():
                    signature_start = prefix_start
            signature = data[signature_start:(body.start_byte if body else node.end_byte)].decode(errors='replace').strip()
            # Function definitions include their exact body separately, not in the signature.
            obj = {
                'name': name, 'qualified': scope + name, 'kind': kind,
                'line': node.start_point.row + 1, 'end_line': node.end_point.row + 1,
                'signature': signature, 'access': access, 'conditions': list(conditions),
                'comment': preceding_comment(node, data), 'members': [], 'parameters': [],
                'calls': [], 'returns': [], 'throws': [],
            }
            if kind == 'type_definition':
                enum = next((child for child in node.named_children if child.type == 'enum_specifier'), None)
                enum_body = enum.child_by_field_name('body') if enum else None
                if enum_body:
                    obj['members'] = [{'declaration': text(child, data), 'access': 'public', 'line': child.start_point.row + 1} for child in enum_body.named_children if child.type == 'enumerator']
            if function:
                params = function.child_by_field_name('parameters')
                obj['parameters'] = [text(n, data) for n in params.named_children] if params else []
                if body:
                    obj['calls'] = list(dict.fromkeys(text(n.child_by_field_name('function'), data) for n in walk(body) if n.type == 'call_expression'))
                    obj['returns'] = list(dict.fromkeys(text(n, data) for n in walk(body) if n.type == 'return_statement'))
                    obj['throws'] = [text(n, data) for n in walk(body) if n.type == 'throw_statement']
            if is_type and body:
                member_access = 'private' if kind == 'class_specifier' else 'public'
                def fields(children):
                    for child in children:
                        if child.type.startswith('preproc_if') or child.type in {'preproc_else', 'preproc_elif'}:
                            yield from fields(child.named_children)
                        else:
                            yield child
                for child in fields(body.named_children):
                    if child.type == 'access_specifier':
                        member_access = text(child, data)
                    elif child.type in {'field_declaration', 'enumerator'}:
                        decl = child.child_by_field_name('declarator')
                        has_function = decl and any(n.type == 'function_declarator' for n in walk(decl))
                        if not has_function:
                            obj['members'].append({'declaration': text(child, data), 'access': member_access, 'line': child.start_point.row + 1})
                record['objects'].append(obj)
                member_access = 'private' if kind == 'class_specifier' else 'public'
                for child in body.named_children:
                    if child.type == 'access_specifier':
                        member_access = text(child, data)
                    else:
                        visit(child, scope + name + '::', member_access, conditions)
                return
            record['objects'].append(obj)
            if is_variable:
                # A declaration can introduce several independent objects.
                for extra in node.children_by_field_name('declarator')[1:]:
                    extra_name = declarator_name(extra, data)
                    record['objects'].append({**obj, 'name': extra_name, 'qualified': scope + extra_name})
            # A function's local variables/lambdas are implementation details, not additional public objects.
            return
        for child in node.named_children:
            visit(child, scope, access, conditions)

    visit(tree.root_node)
    # Preserve bindings and embedded-language objects as their own evidence records.
    from supplements import supplement
    supplement(record, tree, data, root)
    return record
