/* HALO project page — halo-models.js
   Loads the compact mesh bundles written by tools/build_arm_bundle.py and
   tools/build_tracer_bundle.py, and turns their URDF link tree into three.js groups.

   Bundle = <name>.json (links, joints, geometry table) + <name>.bin (uint16 positions,
   int8 normals, uint16/uint32 indices). Coordinates are URDF (x forward, y left, z up);
   the caller places the returned root inside a group rotated -90 deg about X. */
import * as THREE from 'three';

const matCache = new Map();

/** Material from an RGBA colour (sRGB). Neutral mid/light greys become brushed metal. */
export function materialFor(rgba, emissive = false) {
  const key = rgba.join(',') + (emissive ? ':e' : '');
  if (matCache.has(key)) return matCache.get(key);
  const [r, g, b] = rgba;
  const color = new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const sat = Math.max(r, g, b) - Math.min(r, g, b);
  const metal = sat < 0.08 && lum > 0.4 && lum < 0.86;
  const m = new THREE.MeshStandardMaterial({
    color,
    metalness: metal ? 0.78 : 0.04,
    roughness: metal ? 0.3 : lum > 0.85 ? 0.38 : lum < 0.15 ? 0.46 : 0.55,
  });
  if (emissive) { m.emissive = color.clone(); m.emissiveIntensity = 1.4; }
  matCache.set(key, m);
  return m;
}

/** Fetch and decode a bundle. Returns { man, geos: [{ geometry, material }] }. */
export async function loadBundle(baseUrl, name) {
  const man = await (await fetch(`${baseUrl}${name}.json`)).json();
  const res = await fetch(`${baseUrl}${man.bin}`);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${man.bin}`);
  const buf = await res.arrayBuffer();
  const geos = man.geometries.map(g => {
    const q = new Uint16Array(buf, g.pos, g.count * 3);
    const pos = new Float32Array(g.count * 3);
    for (let i = 0; i < g.count; i++) {
      for (let k = 0; k < 3; k++) pos[i * 3 + k] = g.min[k] + (q[i * 3 + k] / 65535) * g.ext[k];
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Int8Array(buf, g.nrm, g.count * 3), 3, true));
    const Idx = g.itype === 'u16' ? Uint16Array : Uint32Array;
    geometry.setIndex(new THREE.BufferAttribute(new Idx(buf, g.idx, g.tris * 3), 1));
    geometry.computeBoundingSphere();
    return { geometry, material: materialFor(g.rgba, !!g.emissive) };
  });
  return { man, geos };
}

const _axis = new THREE.Vector3();

/**
 * Build the link tree of a bundle under `root`.
 * opts.mirrorY  multiply the y of joints whose parent is outside the bundle (mirror an arm mount)
 * opts.skip     link names to leave out
 * Returns { links: {name: Object3D}, set(jointName, value), joints }.
 */
export function buildTree(bundle, root, opts = {}) {
  const links = {}, joints = {};
  const skip = new Set(opts.skip || []);
  for (const [name, e] of Object.entries(bundle.man.links)) {
    if (skip.has(name)) continue;
    const parent = e.parent && links[e.parent] ? links[e.parent] : root;
    const origin = new THREE.Group();
    if (e.xyz) {
      const external = !(e.parent && links[e.parent]);
      origin.position.set(e.xyz[0], e.xyz[1] * (external && opts.mirrorY ? opts.mirrorY : 1), e.xyz[2]);
      origin.quaternion.setFromEuler(new THREE.Euler(e.rpy[0], e.rpy[1], e.rpy[2], 'ZYX'));
    }
    const motion = new THREE.Group();
    motion.name = name;
    origin.add(motion);
    parent.add(origin);
    for (const vi of e.visuals || []) {
      const g = bundle.geos[vi];
      const mesh = new THREE.Mesh(g.geometry, g.material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      motion.add(mesh);
    }
    links[name] = motion;
    if (e.joint && (e.type === 'revolute' || e.type === 'continuous' || e.type === 'prismatic')) {
      joints[e.joint] = { type: e.type, axis: new THREE.Vector3().fromArray(e.axis).normalize(), motion, value: 0 };
    }
  }
  function set(name, v) {
    const j = joints[name];
    if (!j || j.value === v) return;
    j.value = v;
    if (j.type === 'prismatic') j.motion.position.copy(_axis.copy(j.axis).multiplyScalar(v));
    else j.motion.quaternion.setFromAxisAngle(j.axis, v);
  }
  return { links, joints, set };
}
