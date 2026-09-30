"""Evidence for language surfaces nested inside C++ translation units.

Binding descriptions remain authored separately; this pass only records exact
registration expressions, names, conditions and their native targets.
"""
from __future__ import annotations

import re
import json
from functools import lru_cache


@lru_cache(maxsize=1)
def manifest(root):
    return json.loads((root / "docs/reference/supplemental.json").read_text())


def supplement(record, tree, data, root):
    # Macro declarations represent distinct PFN data members, not overloads.
    for obj in record['objects']:
        if obj['name'] == 'GRASSLAND_VULKAN_PROCEDURE_VAR':
            match = re.search(r'GRASSLAND_VULKAN_PROCEDURE_VAR\((\w+)\)', obj['signature'])
            if match:
                obj['qualified'] = obj['qualified'].rsplit('::', 1)[0] + '::' + match.group(1)
                obj['name'] = match.group(1)
                obj['kind'] = 'native_procedure_member'
    extra = manifest(root).get(record['path'])
    if extra:
        if extra['sha256'] != record['sha256']:
            raise ValueError('Stale supplemental objects: ' + record['path'])
        record['objects'].extend(extra['objects'])
    if not record['path'].startswith('code/pybind/'):
        return
    from extract import walk, text
    prefix = 'long_march' if record['path'].endswith('/long_march.cpp') else 'long_march.graphics'
    if record['path'].endswith('/imgui.cpp'):
        prefix += '.imgui'

    def add(name, kind, node, target='', signature=None):
        conditions = []
        parent = node.parent
        while parent:
            if parent.type.startswith('preproc_if'):
                conditions.append(text(parent.child_by_field_name('condition') or parent.child_by_field_name('name'), data))
            parent = parent.parent
        record['objects'].append(dict(name=name.split('.')[-1], qualified=name, kind=kind,
            line=node.start_point.row+1, end_line=node.end_point.row+1,
            signature=signature or text(node,data), access='public', conditions=conditions,
            comment='', members=[], parameters=[], calls=[], returns=[], throws=[], native_target=target))

    for function in [n for n in walk(tree.root_node) if n.type == 'function_definition']:
        variables = {'m': prefix, 'm_graphics': 'long_march.graphics', 'm_imgui': 'long_march.graphics.imgui'}
        # A registrar's classh<T>& parameter determines c's exported class.
        head = text(function, data).split('{',1)[0]
        param = re.search(r'py::classh<([^>]+)>\s*&\s*(\w+)', head)
        if param:
            native, var = param.groups()
            cls = {'Buffer':'DeviceBuffer', 'Core::Settings':'CoreSettings'}.get(native,native)
            variables[var] = 'long_march.graphics.'+cls
        for node in walk(function):
            if node.type == 'declaration':
                match = re.match(r'py::(?:classh|class_|enum_)<(.+?)>\s+(\w+)\s*\(\s*\w+\s*,\s*"([^"]+)"',text(node,data), re.S)
                if match:
                    target,var,name = match.groups();variables[var] = prefix+'.'+name
                    add(variables[var], 'python_type', node, target)
            if node.type != 'call_expression':
                continue
            fn = node.child_by_field_name('function')
            if fn is None or fn.type != 'field_expression':
                continue
            owner = text(fn.child_by_field_name('argument'), data)
            method = text(fn.child_by_field_name('field'), data)
            if owner not in variables or method not in {'def','def_static','def_readwrite','def_readonly','def_property','def_property_readonly','value','attr','def_submodule'}:
                continue
            args = node.child_by_field_name('arguments').named_children
            if not args:continue
            first = text(args[0],data)
            name = first[1:-1] if first.startswith('"') else '__init__' if 'py::init' in first else None
            if name is None:continue
            target = first if name == '__init__' else text(args[1],data) if len(args)>1 else ''
            signature = text(node,data)
            if method == 'attr' and node.parent and node.parent.type=='assignment_expression':
                signature = text(node.parent,data);target=text(node.parent.child_by_field_name('right'),data)
            kind = 'python_enum_value' if method=='value' else 'python_property' if 'property' in method or 'read' in method else 'python_module' if method=='def_submodule' else 'python_constant' if method=='attr' else 'python_callable'
            add(variables[owner]+'.'+name,kind,node,target,signature)
