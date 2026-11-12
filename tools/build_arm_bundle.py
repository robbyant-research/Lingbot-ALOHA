#!/usr/bin/env python3
"""Build the web mesh bundle for the HALO 3D viewer from the ARX X5 robot description.

Input : the `robots/x5/` folder of the aloha_sim2real package
        (robot.urdf, robot.appearance.json, meshes/**.STL).
Output: assets/models/x5_arm.json   kinematic tree + geometry table + materials
        assets/models/x5_arm.bin    quantized vertex / normal / index buffers

What it does
  1. Reads ONE arm (the `left_arm_*` links, both arms share the meshes), the blue head-camera
     bracket and the camera bodies from robot.urdf.
  2. Colours every visual with robot.appearance.json (photo-matched finish). Rules that use
     `components_inside` recolour connected mesh components whose bounds fall in the box.
  3. Decimates each colour group by vertex clustering to a triangle budget, welds vertices,
     splits normals at creases (default 38 deg) and quantizes positions to uint16.

Only numpy + scipy are needed:
  python3 tools/build_arm_bundle.py /path/to/lingbot_sim2real/assets/robots/x5 assets/models
Optional: --budget 0.5 (halve every triangle budget), --crease 38.
"""
import argparse, fnmatch, json, math, os, struct, sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Triangle budget per mesh file (after decimation). Unlisted meshes keep up to 2000 triangles.
BUDGET = {
    'base_link.STL': 2600, 'link1.STL': 1400, 'link2.STL': 6500, 'link3.STL': 4200, 'link4.STL': 1400,
    'link5.STL': 2600, 'motor6_support.STL': 340, 'gripper_body_without_hoop.STL': 3600,
    'long_finger_left.STL': 3200, 'long_finger_right.STL': 3200, 'slider_rack_left.STL': 700,
    'slider_rack_right.STL': 700, 'finger_pad_sleeve.STL': 32, 'camera_mount.STL': 1600,
    'head_camera_mount.STL': 1400, 'camera_pcb.STL': 12, 'camera_lens.STL': 396, 'camera_glass.STL': 192,
    'camera_screws.STL': 200,
}
# Links exported (left arm used as the template; names are stored without the `left_arm_` prefix).
ARM_LINKS = ['left_arm_base_link', 'left_arm_link1', 'left_arm_link2', 'left_arm_link3', 'left_arm_link4',
             'left_arm_link5', 'left_arm_link6', 'left_arm_link7', 'left_arm_link8', 'left_arm_motor6_support',
             'left_arm_camera_mount', 'camera_left_body_approx', 'left_arm_eef_link', 'camera_left_optical_frame']
HEAD_LINKS = ['head_camera_mount', 'camera_head_body_approx', 'camera_head_optical_frame']


def load_stl(path):
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[80:84])[0]
    if 84 + n * 50 == len(b):
        a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
        return a['v'].astype(np.float64)
    pts = [list(map(float, l.split()[1:4])) for l in b.decode('latin1').splitlines() if l.strip().startswith('vertex')]
    return np.array(pts, dtype=np.float64).reshape(-1, 3, 3)


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def weld(tris, tol=1e-7):
    v = tris.reshape(-1, 3)
    key = np.round(v / tol).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    verts = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
    np.add.at(verts, inv, v); np.add.at(cnt, inv, 1)
    return verts / cnt[:, None], inv.reshape(-1, 3)


def components(nv, faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
    _, lab = connected_components(g, directed=False)
    return lab[faces[:, 0]]


def cluster_decimate(verts, faces, target):
    if len(faces) <= target:
        return verts, faces
    span = np.ptp(verts, axis=0).max()
    cell = span / 400.0
    while True:
        key = np.floor((verts - verts.min(0)) / cell).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        nv = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
        np.add.at(nv, inv, verts); np.add.at(cnt, inv, 1); nv /= cnt[:, None]
        f = inv[faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        srt = np.sort(f, axis=1)
        _, keep = np.unique(srt, axis=0, return_index=True)
        f = f[np.sort(keep)]
        if len(f) <= target or cell > span / 20:
            used, f2 = np.unique(f, return_inverse=True)
            return nv[used], f2.reshape(-1, 3)
        cell *= 1.18


def crease_split(verts, faces, crease_deg):
    """Duplicate vertices whose incident faces differ by more than crease_deg; returns verts, normals, faces."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    area = np.linalg.norm(fn, axis=1) + 1e-20
    fu = fn / area[:, None]
    cos_t = math.cos(math.radians(crease_deg))
    corner_v = faces.reshape(-1); corner_f = np.repeat(np.arange(len(faces)), 3)
    order = np.argsort(corner_v, kind='stable')
    out_pos, out_nrm, new_idx = [], [], np.empty(len(corner_v), dtype=np.int64)
    i = 0; n = len(order)
    while i < n:
        j = i
        v = corner_v[order[i]]
        while j < n and corner_v[order[j]] == v:
            j += 1
        groups = []  # [sum_normal, unit_ref, [corners]]
        for k in order[i:j]:
            f = corner_f[k]; nf = fu[f]
            for g in groups:
                if g[1] @ nf > cos_t:
                    g[0] += fn[f]; g[2].append(k); break
            else:
                groups.append([fn[f].copy(), nf, [k]])
        for g in groups:
            idx = len(out_pos)
            out_pos.append(verts[v]); nn = g[0] / (np.linalg.norm(g[0]) + 1e-20); out_nrm.append(nn)
            new_idx[g[2]] = idx
        i = j
    return np.array(out_pos), np.array(out_nrm), new_idx.reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('x5_dir'); ap.add_argument('out_dir')
    ap.add_argument('--budget', type=float, default=1.0); ap.add_argument('--crease', type=float, default=38.0)
    args = ap.parse_args()
    root = ET.parse(os.path.join(args.x5_dir, 'robot.urdf')).getroot()
    app = json.load(open(os.path.join(args.x5_dir, 'robot.appearance.json')))
    palette = {k: v for k, v in app['materials'].items()}

    def find_mesh(fn):
        name = fn.split('/')[-1]
        for d, _, files in os.walk(os.path.join(args.x5_dir, 'meshes')):
            if name in files:
                return os.path.join(d, name)
        raise FileNotFoundError(fn)

    links = {l.get('name'): l for l in root.findall('link')}
    joints = {j.find('child').get('link'): j for j in root.findall('joint')}
    keep = set(ARM_LINKS + HEAD_LINKS)
    geoms, blobs, offset = [], [], 0
    link_out = {}

    def add_blob(arr):
        nonlocal offset
        b = arr.tobytes(); pad = (-len(b)) % 4
        blobs.append(b + b'\0' * pad); start = offset; offset += len(b) + pad
        return start

    for lname in ARM_LINKS + HEAD_LINKS:
        L = links[lname]
        short = lname.replace('left_arm_', '').replace('camera_left_', 'wrist_camera_').replace('camera_head_', 'head_camera_')
        vis_out = []
        for vi, vis in enumerate(L.findall('visual')):
            m = vis.find('geometry/mesh')
            if m is None:
                continue
            fname = m.get('filename').split('/')[-1]
            tris = load_stl(find_mesh(fname))
            o = vis.find('origin')
            if o is not None:
                xyz = np.array([float(t) for t in o.get('xyz', '0 0 0').split()])
                R = rpy_matrix(*[float(t) for t in o.get('rpy', '0 0 0').split()])
                tris = tris @ R.T + xyz
            verts, faces = weld(tris)
            comp = components(len(verts), faces)
            # base colour from the URDF, then appearance rules in order
            mat = vis.find('material'); col = mat.find('color') if mat is not None else None
            base_rgba = [float(t) for t in col.get('rgba').split()] if col is not None else [0.7, 0.7, 0.7, 1]
            face_mat = np.full(len(faces), -1)
            mats = [base_rgba]
            key = f'{lname}/visual_{vi}'
            for rule in app['rules']:
                pat = rule['visual'].replace('[lr]*_', '*_')
                if not fnmatch.fnmatch(key, pat):
                    continue
                rgba = palette[rule['material']]
                if rgba not in mats:
                    mats.append(rgba)
                mi = mats.index(rgba)
                if 'components_inside' not in rule:
                    face_mat[:] = mi
                else:
                    lo, hi = np.array(rule['components_inside'][0]), np.array(rule['components_inside'][1])
                    for c in np.unique(comp):
                        sel = comp == c
                        pts = verts[np.unique(faces[sel])]
                        if np.all(pts.min(0) >= lo - 1e-4) and np.all(pts.max(0) <= hi + 1e-4):
                            face_mat[sel] = mi
            face_mat[face_mat < 0] = 0
            budget_total = int(BUDGET.get(fname, 2000) * args.budget)
            for mi in np.unique(face_mat):
                sub = faces[face_mat == mi]
                share = max(12, int(budget_total * len(sub) / len(faces)))
                used, f2 = np.unique(sub, return_inverse=True)
                v2, f2 = cluster_decimate(verts[used], f2.reshape(-1, 3), share)
                pos, nrm, idx = crease_split(v2, f2, args.crease)
                lo, hi = pos.min(0), pos.max(0); ext = np.maximum(hi - lo, 1e-9)
                q = np.round((pos - lo) / ext * 65535).astype('<u2')
                nq = np.round(nrm * 127).astype('<i1')
                itype = '<u2' if len(pos) < 65536 else '<u4'
                g = {'mesh': fname, 'rgba': [round(x, 4) for x in mats[mi]], 'count': int(len(pos)), 'tris': int(len(idx)),
                     'min': lo.round(6).tolist(), 'ext': ext.round(6).tolist(),
                     'pos': add_blob(q), 'nrm': add_blob(nq), 'idx': add_blob(idx.astype(itype)), 'itype': 'u16' if itype == '<u2' else 'u32'}
                vis_out.append(len(geoms)); geoms.append(g)
        j = joints.get(lname)
        entry = {'visuals': vis_out}
        if j is not None and j.find('parent').get('link') in keep | {'dual_arm_base_link'}:
            o = j.find('origin'); ax = j.find('axis'); lim = j.find('limit')
            entry.update({
                'parent': j.find('parent').get('link').replace('left_arm_', '').replace('camera_left_', 'wrist_camera_').replace('camera_head_', 'head_camera_'),
                'joint': j.get('name').replace('left_arm_', ''), 'type': j.get('type'),
                'xyz': [float(t) for t in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()],
                'rpy': [float(t) for t in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()],
                'axis': [float(t) for t in ax.get('xyz').split()] if ax is not None else [1, 0, 0],
            })
            if lim is not None:
                entry['limit'] = [float(lim.get('lower', 0)), float(lim.get('upper', 0))]
            # Head-camera frames hang off the dual-arm frame, not off one arm: re-express them there.
            if lname in HEAD_LINKS and j.find('parent').get('link') == 'left_arm_base_link':
                m = [float(t) for t in joints['left_arm_base_link'].find('origin').get('xyz').split()]
                entry['parent'] = 'dual_arm_base_link'
                entry['xyz'] = [entry['xyz'][k] + m[k] for k in range(3)]
        link_out[short] = entry
    # the left-arm mount offset is mirrored for the right arm at runtime
    mount = joints['left_arm_base_link'].find('origin').get('xyz')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'x5_arm.bin'), 'wb') as f:
        for b in blobs:
            f.write(b)
    manifest = {
        'source': 'ARX X5 links + HALO custom gripper/cameras from robot.urdf (aloha_sim2real); see README',
        'units': 'metres, URDF frame (x forward, y left, z up)',
        'arm_mount_left_xyz': [float(t) for t in mount.split()],
        'links': link_out, 'geometries': geoms,
        'bin': 'x5_arm.bin', 'bytes': offset,
        'triangles': int(sum(g['tris'] for g in geoms)),
    }
    json.dump(manifest, open(os.path.join(args.out_dir, 'x5_arm.json'), 'w'), indent=1)
    print(f"wrote {len(geoms)} geometries, {manifest['triangles']} triangles, {offset / 1024:.0f} KiB")


if __name__ == '__main__':
    main()
