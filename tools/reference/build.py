#!/usr/bin/env python3
"""Build independent, source-linked architecture and API documents."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

from extract import extract, tracked_files

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'docs/reference'
OUTPUT = ROOT / 'website/reference'


def read_json(name):
    path = SOURCE / name
    return json.loads(path.read_text()) if path.exists() else {}


def esc(value):
    return html.escape(str(value), quote=True)


def paragraphs(items):
    return ''.join(f'<p>{esc(item)}</p>' for item in items)


def bullet_list(items):
    return '<ul>' + ''.join(f'<li>{esc(item)}</li>' for item in items) + '</ul>' if items else ''


def file_url(path):
    return 'files/' + path + '.html'


def module_url(path):
    return 'modules/' + path + '/index.html'


def object_id(obj):
    return 'api-' + hashlib.sha256((obj['qualified'] + obj['signature'] + str(obj['line'])).encode()).hexdigest()[:14]


def api_note(api, path, obj):
    # Shader entry points and file-local helpers may share a spelling without
    # sharing semantics. A file-scoped record always takes precedence.
    return api.get(path + '#' + obj['qualified']) or api.get(obj['qualified'])


def build(strict=False):
    records = [extract(path, ROOT) for path in tracked_files(ROOT)]
    files = read_json('files.json')
    modules = read_json('modules.json')
    api = read_json('api.json')
    paths = {record['path'] for record in records}
    directories = sorted({str(parent) for path in paths for parent in Path(path).parents if str(parent) != '.'})
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    pending = [r['path'] for r in records if r['path'] not in files or files[r['path']].get('sha256') != r['sha256']]
    undocumented = sorted({r['path'] + '#' + o['qualified'] for r in records for o in r['objects'] if not api_note(api, r['path'], o)})
    audits = read_json('diagnostics.json')
    supplemental = read_json('supplemental.json')
    audit_failures = []
    for record in records:
        if record['parse_errors']:
            audit = audits.get(record['path'], {})
            if audit.get('sha256') != record['sha256'] or audit.get('lines') != record['parse_errors'] or not audit.get('review'):
                audit_failures.append(record['path'] + ': parser diagnostics need review')
        raw_starts = [record['source'].count('\n', 0, m.start()) + 1 for m in re.finditer(r'R"[^\s(]*\(', record['source'])]
        if raw_starts and supplemental.get(record['path'], {}).get('raw_string_starts') != raw_starts:
            audit_failures.append(record['path'] + ': embedded source inventory changed')
        if record['path'].startswith('code/pybind/'):
            expected = len(re.findall(r'\.(?:def|def_static|def_readwrite|def_readonly|def_property|def_property_readonly|value|attr|def_submodule)\s*\(', record['source']))
            found = sum(o['kind'].startswith('python_') and o['kind'] != 'python_type' for o in record['objects'])
            if expected != found:
                audit_failures.append(record['path'] + ': Python registration coverage mismatch')
    missing_modules = sorted(set(directories) - set(modules))
    coverage = {'source_revision': revision, 'files': len(records), 'reviewed_files': len(records) - len(pending),
                'objects': sum(len(r['objects']) for r in records), 'documented_object_names': len(api),
                'audit_failures': audit_failures, 'pending_files': pending, 'undocumented_objects': undocumented, 'missing_modules': missing_modules,
                'parser_diagnostics': {r['path']: r['parse_errors'] for r in records if r['parse_errors']}}
    (SOURCE / 'coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + '\n')
    if strict and (pending or undocumented or missing_modules or audit_failures):
        raise SystemExit(f'Reference incomplete: {len(pending)} files, {len(undocumented)} objects, {len(missing_modules)} modules, {len(audit_failures)} source audits')
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    source_base = f'https://github.com/LazyJazzDev/LongMarch/blob/{revision}/'
    symbols = {}
    for record in records:
        for obj in record['objects']:
            symbols.setdefault(obj['qualified'], (file_url(record['path']), object_id(obj)))

    def relative(current, destination):
        return os.path.relpath(OUTPUT / destination, (OUTPUT / current).parent).replace(os.sep, '/')

    def link(current, dest, label, attrs=''):
        return f'<a href="{esc(relative(current, dest))}" {attrs}>{esc(label)}</a>'

    directory_children = {d: [] for d in directories}
    directory_files = {d: [] for d in directories}
    for d in directories:
        parent = str(Path(d).parent)
        if parent in directory_children:
            directory_children[parent].append(d)
    for p in sorted(paths):
        directory_files[str(Path(p).parent)].append(p)

    def navigation(current, current_module):
        def directory(path):
            selected = current_module == path or current_module.startswith(path + '/')
            children = directory_children[path]
            title = modules.get(path, {}).get('title', Path(path).name)
            row = '<details' + (' open' if selected else '') + '><summary>' + esc(title) + '</summary><ul><li>'
            row += link(current, module_url(path), '架构设计与接口约定', 'aria-current="page"' if current == module_url(path) else '') + '</li>'
            for p in directory_files[path]:
                row += '<li>' + link(current, file_url(p), Path(p).name, 'aria-current="page"' if current == file_url(p) else '') + '</li>'
            for child in children:
                row += '<li>' + directory(child) + '</li>'
            return row + '</ul></details>'
        return '<nav aria-label="文档分级目录">' + link(current, 'index.html', '文档首页') + directory('code') + '</nav>'

    def write_page(current, title, content, module='code'):
        path = OUTPUT / current
        path.parent.mkdir(parents=True, exist_ok=True)
        home = os.path.relpath(ROOT / 'website/index.html', path.parent).replace(os.sep, '/')
        css = relative(current, 'reference.css')
        script = relative(current, 'reference.js')
        crumbs = []
        for parent in reversed([module, *[str(p) for p in Path(module).parents if str(p) != '.']]):
            crumbs.append(link(current, module_url(parent), Path(parent).name))
        document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} · LongMarch 文档</title><link rel="stylesheet" href="{css}"><script src="{script}" defer></script></head>
<body><a class="skip" href="#document">跳至正文</a><header class="doc-header"><a class="brand" href="{home}">LongMarch <small>长征 / 文档</small></a><nav aria-label="项目导航"><a href="{home}#home">主页</a><a href="{home}#install">安装和使用</a>{link(current,'index.html','架构说明','aria-current="page"')}<a href="{home}#examples">样例说明</a></nav></header>
<div class="doc-layout"><aside class="doc-sidebar"><label for="filter">筛选目录</label><input id="filter" type="search" placeholder="模块或文件名" autocomplete="off">{navigation(current,module)}</aside>
<main id="document" tabindex="-1"><div class="breadcrumbs">{' / '.join(crumbs)}</div><h1>{esc(title)}</h1>{content}<footer>源码版本 <a href="https://github.com/LazyJazzDev/LongMarch/tree/{revision}">{revision[:10]}</a> · 接口和行为以该版本为准。</footer></main></div></body></html>'''
        path.write_text(document + '\n')

    intro = '<p class="lead">模块设计、逐文件说明与对象接口参考。目录中的每个条目打开独立文档页面。</p>'
    intro += f'<div class="stats"><span>{len(records)} 个源码文件</span><span>{len(directories)} 个目录模块</span><span>{sum(len(r["objects"]) for r in records)} 个声明 / 定义记录</span></div>'
    if pending or undocumented or missing_modules or audit_failures:
        intro += f'<aside class="notice">编写中：已逐文件核对 {len(records)-len(pending)} / {len(records)}；尚有 {len(undocumented)} 个对象名和 {len(missing_modules)} 个模块待补充。源码提取不计作人工审阅。</aside>'
    intro += '<h2>阅读路径</h2><div class="module-cards">'
    for directory in directories:
        if str(Path(directory).parent) == 'code':
            intro += '<article><h3>' + link('index.html',module_url(directory),modules.get(directory,{}).get('title',Path(directory).name)) + '</h3>' + paragraphs([modules.get(directory,{}).get('summary','模块说明正在逐文件整理。')]) + '</article>'
    intro += '</div><h2>接口页包含什么</h2><p>每个文件页说明职责和实现要点，再逐对象列出真实签名、访问权限、条件编译、字段、参数以及对应语义说明。实现中的调用、返回与异常表达式保留为可核查证据。源码正文和精确行号可用于进一步核对。</p>'
    write_page('index.html', '架构与接口文档', intro)
    for directory in directories:
        current = module_url(directory)
        meta = modules.get(directory)
        body = '<p class="path">' + esc(directory) + '</p>'
        if meta:
            body += paragraphs([meta['summary']]) + '<h2>结构与执行流程</h2>' + bullet_list(meta['flow'])
            body += '<h2>接口标准与使用约定</h2>' + bullet_list(meta['contracts'])
        else:
            body += '<aside class="notice">本模块的架构总述尚在逐文件核对，未标记为完整文档。</aside>'
        children = [d for d in directories if str(Path(d).parent) == directory]
        if children:
            body += '<h2>子模块</h2><ul>' + ''.join('<li>' + link(current,module_url(d),modules.get(d,{}).get('title',Path(d).name)) + '</li>' for d in children) + '</ul>'
        body += '<h2>文件职责</h2><div class="file-list">'
        for record in records:
            if str(Path(record['path']).parent) == directory:
                body += '<article><h3>' + link(current,file_url(record['path']),Path(record['path']).name) + '</h3>'
                body += paragraphs([files.get(record['path'],{}).get('summary','待逐文件核对。')]) + '</article>'
        body += '</div>'
        write_page(current, meta['title'] if meta else Path(directory).name, body, directory)

    for record in records:
        path = record['path']; current = file_url(path); note = files.get(path)
        body = f'<p class="path">{esc(path)} · {record["lines"]} 行 · <a href="{source_base}{esc(path)}">查看版本源码 ↗</a></p>'
        if note and note['sha256'] == record['sha256']:
            body += '<h2>职责与设计</h2>' + paragraphs([note['summary']]) + bullet_list(note.get('details',[]))
        else:
            body += '<aside class="notice">本文件尚未完成当前版本的逐项说明，以下签名是源码证据。</aside>'
        audit = audits.get(path)
        if record['parse_errors'] and audit:
            body += '<details class="evidence"><summary>扩展语法核对记录</summary>' + paragraphs([audit['review']]) + '</details>'
        if record['includes']:
            body += '<h2>直接依赖</h2><ul>'
            for include in record['includes']:
                candidates = ['code/' + include, str(Path(path).parent / include)]
                resolved = next((p for p in candidates if p in paths),None)
                body += '<li>' + (link(current,file_url(resolved),include) if resolved else '<code>'+esc(include)+'</code>') + '</li>'
            body += '</ul>'
        if record['objects']:
            body += '<h2>对象与接口</h2><p class="hint">同名重载分别列出；私有成员用于解释实现边界，不构成对外接口。</p>'
        for obj in record['objects']:
            description = api_note(api, record['path'], obj)
            body += f'<article class="api-object" id="{object_id(obj)}"><p class="api-kind">{esc(obj["kind"])} · {esc(obj["access"])} · L{obj["line"]}–{obj["end_line"]}</p><h3>{esc(obj["qualified"])}</h3>'
            body += '<pre><code>' + esc(obj['signature']) + '</code></pre>'
            if description:
                body += paragraphs([description['description']])
                if description.get('contract'):body += '<p class="contract"><strong>约定：</strong>' + esc(description['contract']) + '</p>'
            else:
                body += '<p class="pending">此对象的语义说明尚待核对。</p>'
            if obj['conditions']:body += '<h4>编译条件</h4>' + bullet_list(obj['conditions'])
            if obj['parameters']:body += '<h4>参数声明（顺序与默认值）</h4>' + bullet_list(obj['parameters'])
            if obj['members']:
                body += '<h4>状态与数据字段</h4><div class="table-wrap"><table><thead><tr><th>访问</th><th>声明</th></tr></thead><tbody>'
                body += ''.join('<tr><td>'+esc(m['access'])+'</td><td><code>'+esc(m['declaration'])+'</code></td></tr>' for m in obj['members']) + '</tbody></table></div>'
            if obj['comment']:body += '<h4>源码注释</h4><pre><code>' + esc(obj['comment']) + '</code></pre>'
            if obj['calls'] or obj['returns'] or obj['throws']:
                body += '<details class="evidence"><summary>实现证据：调用、返回与异常</summary><p class="hint">按源码出现顺序列出直接调用表达式；不是完整运行时调用图，也不推断分支一定执行。</p>'
                if obj['calls']:body += '<h4>直接调用</h4>' + bullet_list(obj['calls'])
                if obj['returns']:body += '<h4>返回表达式</h4>' + bullet_list(obj['returns'])
                if obj['throws']:body += '<h4>显式抛出</h4>' + bullet_list(obj['throws'])
                body += '</details>'
            body += f'<a class="source-link" href="{source_base}{esc(path)}#L{obj["line"]}">定位源码 L{obj["line"]} ↗</a></article>'
        body += '<details class="source"><summary>完整文件源码</summary><pre><code>'
        body += '\n'.join(f'<span id="L{i}"><a href="#L{i}" class="line-number">{i}</a> {esc(line)}</span>' for i,line in enumerate(record['source'].splitlines(),1)) + '</code></pre></details>'
        write_page(current, Path(path).name, body, str(Path(path).parent))
    for name in ['reference.css','reference.js']:
        shutil.copyfile(ROOT/'tools/reference'/name, OUTPUT/name)
    print(json.dumps({k:coverage[k] for k in ['files','reviewed_files','objects','documented_object_names']},ensure_ascii=False))


if __name__ == '__main__':
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--strict', action='store_true', help='Reject incomplete or stale manual records')
    build(arguments.parse_args().strict)
