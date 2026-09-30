#!/usr/bin/env python3
"""Validate the generated reference as a navigable static website."""
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2] / 'website'


class Document(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = Counter()
        self.links = []
        self.sidebar = False
        self.sidebar_links = []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids[attrs['id']] += 1
        if tag == 'aside' and attrs.get('class') == 'doc-sidebar':
            self.sidebar = True
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])
        if self.sidebar and tag == 'a':
            self.sidebar_links.append(attrs.get('href', ''))

    def handle_endtag(self, tag):
        if tag == 'aside':
            self.sidebar = False


def main():
    documents = {p.resolve(): Document(p) for p in ROOT.rglob('*.html')}
    errors = []
    count = 0
    for path, document in documents.items():
        if 'reference' not in path.relative_to(ROOT).parts:
            continue
        count += 1
        for ident, repeats in document.ids.items():
            if repeats > 1:
                errors.append(f'{path}: duplicate id {ident}')
        for href in document.sidebar_links:
            url = urlsplit(href)
            if url.fragment or not url.path.endswith('.html'):
                errors.append(f'{path}: sidebar must open an independent page: {href}')
        for href in document.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.is_relative_to(ROOT):
                errors.append(f'{path}: link escapes website: {href}')
            elif not target.exists():
                errors.append(f'{path}: missing target: {href}')
            elif url.fragment and target in documents and unquote(url.fragment) not in documents[target].ids:
                errors.append(f'{path}: missing fragment: {href}')
    if errors:
        raise SystemExit('\n'.join(errors))
    inventory = ROOT.parent / 'docs/reference'
    expected = 1 + len(json.loads((inventory / 'files.json').read_text())) + len(json.loads((inventory / 'modules.json').read_text()))
    if count != expected:
        raise SystemExit(f'Expected {expected} reference documents, got {count}')
    print(f'Validated {count} reference pages: internal links, fragments, IDs and independent sidebar navigation.')


if __name__ == '__main__':
    main()
