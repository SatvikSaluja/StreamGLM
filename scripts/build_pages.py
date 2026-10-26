"""Build a self-contained GitHub Pages evidence site from committed files only."""
import html
import json
import posixpath
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '_site'
PROJECT = json.loads((ROOT / 'results_page/sources/manifest.json').read_text())['project']
REPO = f'https://github.com/SatvikSaluja/{PROJECT}/blob/main/'
STYLE = 'body{max-width:1100px;margin:auto;padding:32px 20px;font:16px/1.65 system-ui;color:#1e2933;background:#fafbf9}a{color:#165e79}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:8px;border:1px solid #dce2e6}pre{padding:16px;background:#edf0ed;overflow:auto}img{max-width:100%}nav{display:flex;gap:20px;flex-wrap:wrap}li{overflow-wrap:anywhere}h1,h2{line-height:1.25}'
TRACKED = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
EXTENSIONS = {'.md', '.json', '.csv', '.txt', '.png', '.svg', '.jpg', '.html', '.css', '.js'}
FILES = {p for p in TRACKED if Path(p).suffix in EXTENSIONS and (
    p.startswith(('results_page/', 'artifacts/', 'results/', 'figures/', 'docs/'))
    or p in {'RESULTS_REPORT.md', 'SIMULATIONS.md', 'RUNBOOK.md', 'constraints-validated.txt'})}
# The main page lives at the site root, preserving existing public source URLs.
def destination(path):
    return path.removeprefix('results_page/')


def rewrite_url(url, source, output):
    parts = urlsplit(html.unescape(url))
    if parts.scheme or parts.netloc or not parts.path or parts.path.startswith('/'):
        return url
    target = posixpath.normpath(posixpath.join(posixpath.dirname(source), unquote(parts.path)))
    suffix = ('?' + parts.query if parts.query else '') + ('#' + parts.fragment if parts.fragment else '')
    if target in FILES:
        dest = destination(target)
        if dest.endswith('.md'):
            dest = dest[:-3] + '.html'
        return posixpath.relpath(dest, posixpath.dirname(output) or '.') + suffix
    # Source-code links remain useful without publishing executable code or caches.
    if target in TRACKED:
        return REPO + target + suffix
    if any(p.startswith(target.rstrip('/') + '/') for p in FILES):
        return posixpath.relpath('evidence.html', posixpath.dirname(output) or '.')
    raise ValueError(f'{source}: missing local link {url!r}')


def rewrite(content, source, output):
    return re.sub(r'\b(href|src)="([^"]*)"', lambda m: m[1] + '="' + html.escape(rewrite_url(m[2], source, output), quote=True) + '"', content)


def wrapper(title, body, output):
    prefix = posixpath.relpath('.', posixpath.dirname(output) or '.')
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title><style>' + STYLE + '</style></head><body><nav aria-label="Results navigation"><a href="' + prefix + '/index.html">Results</a><a href="' + prefix + '/RESULTS_REPORT.html">Written report</a><a href="' + prefix + '/SIMULATIONS.html">Run guide</a><a href="' + prefix + '/evidence.html">All evidence</a></nav><main>' + body + '</main></body></html>'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])


def validate():
    parsed = {}
    for path in OUT.rglob('*.html'):
        doc = Links()
        doc.feed(path.read_text())
        parsed[path.resolve()] = doc
    count = 0
    for path, doc in parsed.items():
        for link in doc.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not target.is_relative_to(OUT.resolve()) or not target.is_file():
                raise ValueError(f'{path.relative_to(OUT)}: broken link {link}')
            if parts.fragment and target in parsed and unquote(parts.fragment) not in parsed[target].ids:
                raise ValueError(f'{path.relative_to(OUT)}: missing anchor {link}')
            count += 1
    print(f'Validated {len(parsed)} HTML pages and {count} internal links')


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for source in sorted(FILES):
        output = destination(source)
        target = OUT / output
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
        if source.endswith('.md'):
            output = output[:-3] + '.html'
            body = markdown.markdown((ROOT / source).read_text(), extensions=['tables', 'fenced_code', 'toc'])
            (OUT / output).write_text(wrapper(Path(source).stem, rewrite(body, source, output), output))
        elif source.endswith('.html'):
            content = rewrite(target.read_text(), source, output)
            if output == 'index.html':
                content = content.replace('</nav>', '<a href="evidence.html">All evidence</a></nav>', 1)
            target.write_text(content)
    links = []
    for source in sorted(FILES):
        if source == 'results_page/index.html':
            continue
        target = destination(source)
        if target.endswith('.md'):
            target = target[:-3] + '.html'
        links.append(f'<li><a href="{html.escape(target)}">{html.escape(source)}</a></li>')
    body = '<h1>' + html.escape(PROJECT) + ' evidence archive</h1><p>All published reports, figures and compact evidence files from this repository. Historical, incomplete and failed runs retain their original labels; inclusion here does not establish a successful result. Large recordings and model caches are excluded.</p><ul>' + ''.join(links) + '</ul>'
    (OUT / 'evidence.html').write_text(wrapper(PROJECT + ' evidence', body, 'evidence.html'))
    (OUT / '.nojekyll').touch()
    validate()
    print(f'Built {PROJECT}: {len(FILES)} source files in {OUT}')


if __name__ == '__main__':
    main()
