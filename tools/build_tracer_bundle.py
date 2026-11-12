#!/usr/bin/env python3
"""Build the web mesh bundle for the AgileX Tracer 2.0 base used by the HALO 3D viewer.

Input : AgileX `tracer2_description` folder (urdf/tracer2_description.urdf + meshes/*.STL), from
        https://github.com/agilexrobotics/ugv_gazebo_sim  (branch master, folder tracer/tracer2_description)
Output: assets/models/tracer2.json + assets/models/tracer2.bin (same format as x5_arm.*)

The official meshes carry no colours, so the body is split into connected components and coloured
to match the photo of the real robot (white shell, graphite deck, aluminium deck rails, light bar).

  python3 tools/build_tracer_bundle.py /path/to/tracer2_description assets/models
"""
import argparse, json, os, sys
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_arm_bundle import load_stl, weld, components, cluster_decimate, crease_split  # noqa: E402

COL = {
    'shell': [0.90, 0.91, 0.92, 1], 'deck': [0.14, 0.15, 0.16, 1], 'trim': [0.08, 0.085, 0.09, 1],
    'rail': [0.74, 0.76, 0.78, 1], 'light': [0.86, 0.95, 1.0, 1], 'rear_light': [0.45, 0.06, 0.05, 1],
    'tire': [0.11, 0.115, 0.12, 1], 'hub': [0.55, 0.58, 0.61, 1], 'caster': [0.22, 0.23, 0.25, 1],
}


def classify_body(verts, faces):
    """Return a colour key per face of the Tracer 2 body."""
    lab = components(len(verts), faces)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
    out = np.empty(len(faces), dtype=object)
    for k in np.unique(lab):
        sel = lab == k
        p = verts[np.unique(faces[sel])]
        lo, hi = p.min(0), p.max(0); size = hi - lo; ctr = (lo + hi) / 2
        if size[0] > 0.45 and size[1] < 0.06 and lo[2] > -0.01:
            out[sel] = 'rail'                                   # deck mounting rails
        elif size[0] < 0.03 and size[1] < 0.06 and lo[2] > -0.01:
            out[sel] = 'trim'                                   # rail end blocks
        elif ctr[0] > 0.28 and size[1] > 0.15 and size[1] < 0.22 and size[2] < 0.02:
            out[sel] = 'light'                                  # front light bar
        elif ctr[0] < -0.34 and size[1] < 0.15 and size[2] < 0.03:
            out[sel] = 'rear_light'
        elif size[0] > 0.18 and size[1] > 0.5:
            up = n[sel][:, 2] > 0.55                            # shell: graphite top, white sides
            out[np.where(sel)[0][up]] = 'deck'
            out[np.where(sel)[0][~up]] = 'shell'
        elif size[0] > 0.2 and size[1] > 0.15 and size[2] < 0.01:
            out[sel] = 'deck'                                   # top cover plates
        else:
            out[sel] = 'trim'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('desc_dir'); ap.add_argument('out_dir'); ap.add_argument('--crease', type=float, default=35.0)
    args = ap.parse_args()
    root = ET.parse(os.path.join(args.desc_dir, 'urdf', 'tracer2_description.urdf')).getroot()
    blobs, offset, geoms = [], 0, []

    def add_blob(arr):
        nonlocal offset
        b = arr.tobytes(); pad = (-len(b)) % 4
        blobs.append(b + b'\0' * pad); s = offset; offset += len(b) + pad
        return s

    def emit(verts, faces, rgba, budget, name):
        v2, f2 = cluster_decimate(verts, faces, budget)
        pos, nrm, idx = crease_split(v2, f2, args.crease)
        lo, hi = pos.min(0), pos.max(0); ext = np.maximum(hi - lo, 1e-9)
        itype = '<u2' if len(pos) < 65536 else '<u4'
        geoms.append({'mesh': name, 'rgba': rgba, 'count': int(len(pos)), 'tris': int(len(idx)),
                      'min': lo.round(6).tolist(), 'ext': ext.round(6).tolist(),
                      'pos': add_blob(np.round((pos - lo) / ext * 65535).astype('<u2')),
                      'nrm': add_blob(np.round(nrm * 127).astype('<i1')),
                      'idx': add_blob(idx.astype(itype)), 'itype': 'u16' if itype == '<u2' else 'u32',
                      'emissive': rgba == COL['light']})
        return len(geoms) - 1

    mesh = lambda n: os.path.join(args.desc_dir, 'meshes', n)
    parts = {}
    # body
    v, f = weld(load_stl(mesh('base_link.STL')))
    cls = classify_body(v, f)
    budget = {'shell': 5200, 'deck': 2400, 'trim': 2600, 'rail': 300, 'light': 200, 'rear_light': 120}
    ids = []
    for key in budget:
        sel = cls == key
        if sel.any():
            used, ff = np.unique(f[sel], return_inverse=True)
            ids.append(emit(v[used], ff.reshape(-1, 3), COL[key], budget[key], f'base_link.STL:{key}'))
    parts['base_link'] = {'visuals': ids}
    # wheels: tyre / hub split by radius around the local z axle
    for link in ['left_Link', 'right_link']:
        v, f = weld(load_stl(mesh(link + '.STL')))
        cen = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3
        r = np.hypot(cen[:, 0], cen[:, 1])
        tyre = r > 0.047
        vis = []
        for sel, key, bud in ((tyre, 'tire', 2400), (~tyre, 'hub', 1600)):
            used, ff = np.unique(f[sel], return_inverse=True)
            vis.append(emit(v[used], ff.reshape(-1, 3), COL[key], bud, f'{link}.STL:{key}'))
        parts[link] = {'visuals': vis}
    for link in ['left_front_link', 'right_front_link']:
        v, f = weld(load_stl(mesh(link + '.STL')))
        parts[link] = {'visuals': [emit(v, f, COL['caster'], 1400, link + '.STL')]}
    parts['left_rear_link'] = {'visuals': parts['left_front_link']['visuals']}
    parts['right_rear_link'] = {'visuals': parts['right_front_link']['visuals']}
    for j in root.findall('joint'):
        child = j.find('child').get('link'); o = j.find('origin'); ax = j.find('axis')
        if child in parts:
            parts[child].update({'parent': j.find('parent').get('link'), 'joint': j.get('name'), 'type': j.get('type'),
                                 'xyz': [float(t) for t in o.get('xyz').split()], 'rpy': [float(t) for t in o.get('rpy').split()],
                                 'axis': [float(t) for t in ax.get('xyz').split()] if ax is not None else [0, 0, 1]})
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'tracer2.bin'), 'wb') as fh:
        for b in blobs:
            fh.write(b)
    manifest = {'source': 'AgileX Tracer 2.0, agilexrobotics/ugv_gazebo_sim tracer/tracer2_description (package.xml license: BSD); colours added',
                'units': 'metres, URDF frame (x forward, y left, z up); base_link is 0.153 m above the ground',
                'ground_offset': 0.153, 'links': parts, 'geometries': geoms, 'bin': 'tracer2.bin', 'bytes': offset,
                'triangles': int(sum(g['tris'] for g in geoms))}
    json.dump(manifest, open(os.path.join(args.out_dir, 'tracer2.json'), 'w'), indent=1)
    print(f"wrote {len(geoms)} geometries, {manifest['triangles']} triangles, {offset / 1024:.0f} KiB")


if __name__ == '__main__':
    main()
