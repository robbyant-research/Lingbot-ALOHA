"""Bundle the HALO viewer and its vendored ES modules for local HTML use."""
from pathlib import Path
import base64
import json
import re
import subprocess
import shutil

SITE = Path(__file__).resolve().parents[1]

def read(name):
    return (SITE/name).read_text(encoding='utf-8')

def strip_imports(source):
    return re.sub(r'^import\s+.*?\s+from\s+[\'\"][^\'\"]+[\'\"];\s*', '', source, flags=re.MULTILINE|re.DOTALL)

def addon(name, exported):
    source = read('vendor/three/addons/'+name)
    imports = re.search(r'import\s*\{(.*?)\}\s*from\s*[\'\"]three[\'\"];',source,re.DOTALL)
    assert imports, name
    keys = imports.group(1)
    source = source[:imports.start()] + 'const {' + keys + '} = THREE;' + source[imports.end():]
    source = re.sub(r'export\s*\{\s*'+exported+r'\s*\};', '', source)
    assert not re.search(r'^\s*(import|export)\b',source,re.MULTILINE),name
    return 'const {'+exported+'} = (function(){\n'+source+'\nreturn {'+exported+'};\n})();\n'

three = read('vendor/three/three.module.min.js')
export = re.search(r'export\s*\{([^}]+)\};?\s*$',three)
assert export
exports = []
for item in export.group(1).split(','):
    parts = re.split(r'\s+as\s+',item.strip())
    assert len(parts) in (1,2),item
    original, public = (parts[0],parts[-1])
    assert re.fullmatch(r'[\w$]+',original) and re.fullmatch(r'[\w$]+',public),item
    exports.append(public+':'+original)
three = three[:export.start()]+'\nreturn {'+','.join(exports)+'};\n'

bundles = {}
for name in ['x5_arm','tracer2']:
    manifest = json.loads(read('assets/models/'+name+'.json'))
    binary = (SITE/'assets/models'/manifest['bin']).read_bytes()
    bundles[name] = dict(man=manifest,bin=base64.b64encode(binary).decode('ascii'))

models = strip_imports(read('halo-models.js'))
models = re.sub(r'\bexport\s+(?=(?:async\s+)?function\b)', '', models)
old = '''  const man = await (await fetch(`${baseUrl}${name}.json`)).json();
  const res = await fetch(`${baseUrl}${man.bin}`);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${man.bin}`);
  const buf = await res.arrayBuffer();'''
new = '''  const embedded = embeddedBundles[name];
  if (!embedded) throw new Error(`Missing embedded HALO model: ${name}`);
  const man = embedded.man;
  const decoded = atob(embedded.bin);
  const bytes = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; i++) bytes[i] = decoded.charCodeAt(i);
  const buf = bytes.buffer;'''
assert old in models
models = models.replace(old,new)
main = strip_imports(read('halo3d.js'))
assert not re.search(r'^\s*(import|export)\b',models+main,re.MULTILINE)
bundle = '''/* HALO local viewer. Generated from the ORIGINAL unchanged three.js r169,
 * OrbitControls, RoomEnvironment, halo-models.js and halo3d.js.
 * Model binaries are embedded to support double-clicking index.html.
 * Original animation preserved; ICR removed and frame fittings corrected at the user's request.
 * three.js and its addons: MIT, Copyright Three.js Authors. */
(function(){
'use strict';
const THREE=(function(){
'''+three+'})();\n'+addon('OrbitControls.js','OrbitControls')+addon('RoomEnvironment.js','RoomEnvironment')
bundle += 'const {loadBundle,buildTree}=(function(){\nconst embeddedBundles='+json.dumps(bundles,separators=(',',':'))+';\n'+models+'\nreturn {loadBundle,buildTree};\n})();\n'
frame_parts = read('frame-parts.js').replace('export const HALO_FRAME_PARTS', 'const HALO_FRAME_PARTS')
bundle += '(function(){\n'+frame_parts+'\n'+main+'\n})();\n})();\n'
(SITE/'halo-viewer.js').write_text(bundle,encoding='utf-8')
if shutil.which('node'):
    check = subprocess.run(['node','--check',str(SITE/'halo-viewer.js')],text=True,capture_output=True)
    assert check.returncode==0,check.stderr

html = read('index.html')
html = re.sub(r'  <!-- three\.js r169.*?<script type="module" src="halo3d\.js\?v=[^\"]+"></script>',
              '  <!-- The original HALO animation, bundled with local models for file:// and HTTP. -->\n  <script src="halo-viewer.js?v=20261013local" defer></script>',html,flags=re.DOTALL)
html = re.sub(r'halo-viewer\.js\?v=[^"\s]+', 'halo-viewer.js?v=20261105palmcenter', html)
assert 'halo-viewer.js?v=20261105palmcenter' in html and 'type="module"' not in html
(SITE/'index.html').write_text(html,encoding='utf-8')
print(f'HALO animation bundled for local HTML: {len(bundle.encode("utf-8")):,} bytes.')
