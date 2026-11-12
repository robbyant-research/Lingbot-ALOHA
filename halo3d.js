/* HALO project page — halo3d.js
   Scroll-driven three.js viewer of the HALO hardware (ES module).

   Models
     - ARX X5 arms with HALO's gripper, wrist cameras and head camera: assets/models/x5_arm.*
       (built by tools/build_arm_bundle.py from the robot description package)
     - AgileX Tracer 2.0 base: assets/models/tracer2.* (tools/build_tracer_bundle.py, official AgileX meshes)
     - Aluminium framework: original outer dimensions retained (880 x 710 mm).
       Brace anchors, 45-degree fittings and three-way nodes follow the supplied Blender scene
       via frame-parts.js. Green encoder measuring wheels follow the supplied hardware reference.
   three.js r169 is vendored in vendor/three/ and resolved through the import map in index.html.

   Markup hooks (see the "mechanical design" module in index.html):
     [data-mech] wrapper (gets .is-live once the models are loaded), .mech__canvas, .mech__labels,
     .mech__leaders, .mech-step[data-step], .mech__dot, [data-readout], [data-caption], .mech__loading.
   Without WebGL, or if the model files cannot be fetched (e.g. opened via file://), the Fig. 4 photo
   inside the stage stays visible. */

import * as THREE from 'three';
import { OrbitControls } from './vendor/three/addons/OrbitControls.js';
import { RoomEnvironment } from './vendor/three/addons/RoomEnvironment.js';
import { loadBundle, buildTree } from './halo-models.js?v=20261012b';
import { HALO_FRAME_PARTS } from './frame-parts.js';

const MODELS = 'assets/models/';
const SHOW_OPERATOR = true;   // translucent operator in the data-collection steps

const rootEl = document.querySelector('[data-mech]');
if (rootEl) init(rootEl).catch(err => {
  console.warn('[HALO 3D] viewer disabled:', err);
  rootEl.classList.remove('is-live');
  rootEl.classList.add('is-failed');
});

async function init(root) {
  /* ------------------------------------------------------------------ DOM */
  const stage = root.querySelector('.mech__stage');
  const host = root.querySelector('.mech__canvas');
  const svg = root.querySelector('.mech__leaders');
  const labelLayer = root.querySelector('.mech__labels');
  const readout = root.querySelector('[data-readout]');
  const captionEl = root.querySelector('[data-caption]');
  const hint = root.querySelector('.mech__hint');
  const dots = Array.from(root.querySelectorAll('.mech__dot'));
  const steps = Array.from(root.querySelectorAll('.mech-step'));
  if (!stage || !host || !steps.length) return;
  const probe = document.createElement('canvas');
  if (!(probe.getContext('webgl2') || probe.getContext('webgl'))) return;

  const mqReduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  const mqDesktop = window.matchMedia('(min-width: 1024px)');
  const mqFine = window.matchMedia('(hover: hover) and (pointer: fine)');
  let reduce = mqReduce.matches;

  root.classList.add('is-loading');
  const bundlesP = Promise.all([loadBundle(MODELS, 'x5_arm'), loadBundle(MODELS, 'tracer2')]);

  /* ------------------------------------------------------------------ math helpers */
  const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
  const V2 = (x, y) => new THREE.Vector2(x, y);
  const UP = V(0, 1, 0), ZAX = V(0, 0, 1);
  const clamp = THREE.MathUtils.clamp, lerp = THREE.MathUtils.lerp, DEG = Math.PI / 180;
  const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
  const lerpAngle = (a, b, k) => a + wrap(b - a) * k;
  const _a = V(), _b = V(), _c = V(), _d = V(), _e = V();

  /* ------------------------------------------------------------------ renderer / scene */
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  host.appendChild(renderer.domElement);

  const BG = 0xe6e9ed;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);
  scene.fog = new THREE.Fog(BG, 7.5, 17);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.85;
  const camera = new THREE.PerspectiveCamera(35, 1, 0.02, 60);

  scene.add(new THREE.HemisphereLight(0xffffff, 0xc4c9d1, 0.8));
  const sun = new THREE.DirectionalLight(0xffffff, 1.8);
  const SUN_OFF = V(2.4, 6, 3.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -2.2, right: 2.2, top: 2.2, bottom: -2.2, near: 1, far: 14 });
  sun.shadow.bias = -0.0003;
  sun.shadow.normalBias = 0.015;
  sun.shadow.radius = 3;
  scene.add(sun, sun.target);
  const fill = new THREE.DirectionalLight(0xffffff, 0.35);
  fill.position.set(-3, 2.5, -2);
  scene.add(fill);

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.ShadowMaterial({ opacity: 0.16 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(24, 48, 0xc2c7ce, 0xd3d7dc);
  grid.material.transparent = true;
  grid.material.opacity = 0.75;
  grid.position.y = 0.001;
  scene.add(grid);

  /* ------------------------------------------------------------------ materials */
  const M = (color, o) => new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.5, metalness: 0 }, o));
  const mat = {
    alu: M(0xd3d8de, { metalness: 0.82, roughness: 0.34 }),
    aluDark: M(0xa3aab2, { metalness: 0.7, roughness: 0.36 }),
    cap: M(0x1b1d21, { roughness: 0.6 }),
    plate: M(0x17191c, { roughness: 0.45, metalness: 0.3 }),
    steel: M(0xb7bcc2, { metalness: 0.9, roughness: 0.25 }),
    tyre: M(0x6f747a, { roughness: 0.82 }),          // grey rubber casters (Fig. 4)
    hub: M(0xc9cdd2, { metalness: 0.85, roughness: 0.28 }),
    enc: M(0x1f2226, { roughness: 0.4, metalness: 0.35 }),
    encoderTyre: M(0x6eaa37, { roughness: 0.72 }),    // green measuring-wheel tread
    encRing: M(0x9d68f4, { roughness: 0.4 }),         // purple ring = encoder colour in Fig. 4
    fixedRing: M(0x00b0f0, { roughness: 0.4 }),       // cyan ring = fixed-direction wheel colour in Fig. 4
  };

  /* ------------------------------------------------------------------ geometry helpers */
  const CYL = new THREE.CylinderGeometry(1, 1, 1, 32);
  const SPH = new THREE.SphereGeometry(1, 28, 18);
  const BOX = new THREE.BoxGeometry(1, 1, 1);
  const PROFILE = (() => { // 40 x 40 mm T-slot extrusion with centre bore, length 1 along z
    const s = 0.02, n = 0.0041, d = 0.0042, l = 0.0068, sh = new THREE.Shape();
    const side = (fn) => { fn(-s, -s); fn(-n, -s); fn(-n, -s + d * 0.45); fn(-l, -s + d * 0.45); fn(-l, -s + d); fn(l, -s + d); fn(l, -s + d * 0.45); fn(n, -s + d * 0.45); fn(n, -s); };
    const pts = [];
    const rot = (k) => (x, y) => { const c = Math.cos(k * Math.PI / 2), si = Math.sin(k * Math.PI / 2); pts.push(V2(x * c - y * si, x * si + y * c)); };
    for (let k = 0; k < 4; k++) side(rot(k));
    sh.setFromPoints(pts);
    const hole = new THREE.Path(); hole.absarc(0, 0, 0.0035, 0, Math.PI * 2, true); sh.holes.push(hole);
    const g = new THREE.ExtrudeGeometry(sh, { depth: 1, bevelEnabled: false, curveSegments: 10 });
    g.translate(0, 0, -0.5);
    return g;
  })();
  function mesh(geo, m, parent, shadow = true) {
    const o = new THREE.Mesh(geo, m);
    o.castShadow = shadow; o.receiveShadow = true;
    if (parent) parent.add(o);
    return o;
  }
  function box(parent, min, max, m) {
    const o = mesh(BOX, m, parent);
    o.scale.set(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    o.position.set((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    return o;
  }
  function cyl(parent, c, r, h, axis, m) {
    const o = mesh(CYL, m, parent);
    o.scale.set(r, h, r);
    o.position.copy(c);
    if (axis === 'x') o.rotation.z = Math.PI / 2; else if (axis === 'z') o.rotation.x = Math.PI / 2;
    return o;
  }
  function between(o, a, b, r) {
    _a.subVectors(b, a);
    const L = _a.length() || 1e-6;
    o.position.addVectors(a, b).multiplyScalar(0.5);
    o.scale.set(r, L, r);
    o.quaternion.setFromUnitVectors(UP, _a.divideScalar(L));
  }
  function rail(parent, a, b, m = mat.alu, k = 1) {
    const o = mesh(PROFILE, m, parent);
    _a.subVectors(b, a);
    o.scale.set(k, k, _a.length());
    o.position.addVectors(a, b).multiplyScalar(0.5);
    o.quaternion.setFromUnitVectors(ZAX, _a.normalize());
    return o;
  }
  function endCap(parent, p, dir) { // thin black cap on an open profile end
    const o = box(parent, [-0.0205, -0.0205, -0.002], [0.0205, 0.0205, 0.002], mat.cap);
    o.position.copy(p);
    o.quaternion.setFromUnitVectors(ZAX, dir);
    return o;
  }
  const GUSSET = (() => { // 40 mm corner bracket (triangular plate with a lip)
    const sh = new THREE.Shape([V2(0, 0), V2(0.038, 0), V2(0, 0.038)]);
    return new THREE.ExtrudeGeometry(sh, { depth: 0.004, bevelEnabled: false });
  })();
  function gusset(parent, corner, ux, uy) { // plate in the plane spanned by ux, uy at `corner`
    const o = mesh(GUSSET, mat.aluDark, parent);
    const n = _b.crossVectors(ux, uy).normalize();
    o.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(ux, uy, n));
    o.position.copy(corner);
    return o;
  }
  function tyreGeometry(R, W, inner) {
    const pts = [], rr = W * 0.32;
    pts.push(V2(inner, -W / 2));
    for (let i = 0; i <= 6; i++) { const t = -Math.PI / 2 + (i / 6) * (Math.PI / 2); pts.push(V2(R - rr + rr * Math.cos(t), -W / 2 + rr + rr * Math.sin(t))); }
    for (let i = 0; i <= 6; i++) { const t = (i / 6) * (Math.PI / 2); pts.push(V2(R - rr + rr * Math.cos(t), W / 2 - rr + rr * Math.sin(t))); }
    pts.push(V2(inner, W / 2));
    const g = new THREE.LatheGeometry(pts, 40);
    g.rotateX(Math.PI / 2);   // axle along z
    return g;
  }
  const TYRE = tyreGeometry(0.05, 0.032, 0.03);
  function makeWheel(ringMat) { // grey castor wheel, axle along z; returns { g, spin }
    const g = new THREE.Group(), spin = new THREE.Group();
    g.add(spin);
    mesh(TYRE, mat.tyre, spin);
    cyl(spin, V(0, 0, 0), 0.031, 0.026, 'z', mat.hub);
    for (const s of [-1, 1]) cyl(spin, V(0, 0, s * 0.0135), 0.022, 0.002, 'z', ringMat || mat.aluDark);
    for (let i = 0; i < 4; i++) { // hub bolts make the rotation readable
      const a = (i / 4) * Math.PI * 2;
      cyl(spin, V(Math.cos(a) * 0.014, Math.sin(a) * 0.014, 0.0145), 0.003, 0.004, 'z', mat.cap);
    }
    cyl(g, V(0, 0, 0), 0.006, 0.05, 'z', mat.steel);     // axle bolt (does not spin)
    return { g, spin, angle: 0, r: 0.05 };
  }
  function makeFrustum(len, hfov, vfov, color) { // camera frustum along +z (optical frame)
    const hw = len * Math.tan(hfov / 2), hh = len * Math.tan(vfov / 2);
    const P = [[hw, hh, len], [-hw, hh, len], [-hw, -hh, len], [hw, -hh, len]];
    const pts = [];
    P.forEach(p => pts.push(0, 0, 0, ...p));
    for (let i = 0; i < 4; i++) pts.push(...P[i], ...P[(i + 1) % 4]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    const lm = new THREE.LineBasicMaterial({ color, transparent: true, depthWrite: false });
    const pm = new THREE.MeshBasicMaterial({ color, transparent: true, side: THREE.DoubleSide, depthWrite: false });
    lm.userData.baseOpacity = 0.95; pm.userData.baseOpacity = 0.13;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(2 * hw, 2 * hh), pm);
    plane.position.z = len;
    const g = new THREE.Group();
    g.add(new THREE.LineSegments(geo, lm), plane);
    g.userData.mats = [lm, pm];
    return g;
  }
  function dashedLine(points, color, dash, gap, parent) {
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const m = new THREE.LineDashedMaterial({ color, dashSize: dash, gapSize: gap, transparent: true, depthWrite: false });
    const l = new THREE.Line(geo, m);
    l.computeLineDistances();
    if (parent) parent.add(l);
    return l;
  }
  function setOpacity(list, o) {
    for (const m of list) {
      const t = o < 0.999;
      if (m.transparent !== t) { m.transparent = t; m.needsUpdate = true; }
      m.opacity = o * (m.userData.baseOpacity ?? 1);
      m.depthWrite = !t || o > 0.6;
    }
  }
  // Two-bone IK used for the operator's limbs.
  function twoBone(S, H, a, b, pole, out) {
    _b.subVectors(H, S);
    const D = _b.length() || 1e-6;
    const Dc = clamp(D, Math.abs(a - b) + 1e-3, a + b - 1e-3);
    _b.divideScalar(D);
    const cosA = clamp((a * a + Dc * Dc - b * b) / (2 * a * Dc), -1, 1);
    _c.copy(pole).addScaledVector(_b, -pole.dot(_b)).normalize();
    return out.copy(S).addScaledVector(_b, a * cosA).addScaledVector(_c, a * Math.sqrt(1 - cosA * cosA));
  }

  /* ------------------------------------------------------------------ framework (880 x 710 mm) */
  // three.js frame of the platform: x forward, y up, z to the robot's right. Floor at y = 0.
  const platform = new THREE.Group();
  scene.add(platform);
  const TOP = 0.90, TOPC = TOP - 0.02, FX = 0.42, FZ = 0.335, BOTC = 0.135, FOOT = 0.115;
  const frameG = new THREE.Group();
  platform.add(frameG);
  // Keep the original outer dimensions. Split rails at the reference CAD three-way nodes.
  for (const z of [-FZ, FZ]) {
    rail(frameG, V(-0.44, TOPC, z), V(-0.02, TOPC, z));
    rail(frameG, V(0.02, TOPC, z), V(FX - 0.02, TOPC, z));
  }
  for (const x of [-FX, FX]) rail(frameG, V(x, TOPC, -FZ + 0.02), V(x, TOPC, FZ - 0.02));
  // lower body (rear half): mid posts on the drive-wheel axis (x = 0) and rear posts
  for (const x of [0, -FX]) for (const z of [-FZ, FZ]) rail(frameG, V(x, BOTC + 0.02, z), V(x, TOPC - 0.02, z));
  for (const z of [-FZ, FZ]) {
    rail(frameG, V(-0.47, BOTC, z), V(-FX - 0.02, BOTC, z));                 // retain rear outrigger
    rail(frameG, V(-FX + 0.02, BOTC, z), V(-0.02, BOTC, z));
  }
  rail(frameG, V(-FX, BOTC, -FZ + 0.02), V(-FX, BOTC, FZ - 0.02));
  // Move the two rear braces into the rear cross-frame, along the user's marked diagonals.
  // Rotate each complete brace and its end fittings about the corresponding upright.
  function relocateRearBrace(member) {
    const side = Math.sign(member.position.z), pivot = V(-FX, 0, side * FZ);
    const rotation = new THREE.Matrix4().makeRotationY(side * Math.PI / 2);
    rotation.setPosition(pivot.clone().sub(pivot.clone().applyMatrix4(rotation)));
    member.applyMatrix4(rotation);
  }
  // Reference braces stop at separate 45-degree end-face fittings instead of intersecting the rails.
  for (const brace of HALO_FRAME_PARTS.braces) {
    const member = rail(frameG, V(...brace.endpoints[0]), V(...brace.endpoints[1]));
    member.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(...brace.basis.map(b => V(...b))));
    if (brace.name.startsWith('Rear ')) relocateRearBrace(member);
  }
  const fittingGeometries = new Map();
  for (const [name, part] of Object.entries(HALO_FRAME_PARTS.templates)) {
    const decoded = atob(part.buffer), bytes = new Uint8Array(decoded.length);
    for (let i = 0; i < decoded.length; i++) bytes[i] = decoded.charCodeAt(i);
    const geo = new THREE.BufferGeometry(), n = part.count * 3;
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(bytes.buffer, 0, n), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(bytes.buffer, n * 4, n), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(bytes.buffer, n * 8, part.indexCount), 1));
    fittingGeometries.set(name, geo);
  }
  for (const part of HALO_FRAME_PARTS.instances) {
    const fitting = mesh(fittingGeometries.get(part.geometry), mat[part.material], frameG);
    fitting.name = part.name;
    fitting.position.set(...part.position);
    if (/^Rear (lower|upper) end-face/.test(part.name)) relocateRearBrace(fitting);
  }
  // head-camera mast on the front rail (position from the robot description: 55 mm ahead of the arm bases)
  rail(frameG, V(FX, TOP, 0), V(FX, TOP + 0.2, 0));
  endCap(frameG, V(FX, TOP + 0.202, 0), UP);
  // end caps and corner brackets
  for (const z of [-FZ, FZ]) {
    endCap(frameG, V(-0.442, TOPC, z), V(-1, 0, 0));
    endCap(frameG, V(-0.472, BOTC, z), V(-1, 0, 0));
  }
  // black arm-mount plates at the two front corners
  const ARM_X = 0.365, ARM_Y = TOP + 0.01, ARM_Z = 0.3146;
  for (const s of [-1, 1]) box(frameG, [ARM_X - 0.071, TOP, s * ARM_Z - 0.062], [ARM_X + 0.071, ARM_Y, s * ARM_Z + 0.062], mat.plate);

  /* ------------------------------------------------------------------ wheels */
  // Passive swivel casters under the rear posts.
  const casters = [];
  for (const z of [-FZ, FZ]) {
    const g = new THREE.Group();
    g.position.set(-FX, FOOT, z);
    box(g, [-0.034, -0.005, -0.034], [0.034, 0, 0.034], mat.hub);
    const sw = new THREE.Group(); g.add(sw);
    cyl(sw, V(0, -0.011, 0), 0.03, 0.012, 'y', mat.hub);
    box(sw, [-0.016, -0.019, -0.023], [0.012, -0.015, 0.023], mat.hub);
    for (const s of [-1, 1]) {
      const leg = box(sw, [-0.012, -0.05, -0.0025], [0.012, 0.0, 0.0025], mat.hub);
      leg.position.set(-0.012, -0.043, s * 0.0215); leg.rotation.z = -0.32;
    }
    const w = makeWheel();
    w.g.position.set(-0.026, 0.05 - FOOT, 0);
    sw.add(w.g);
    platform.add(g);
    casters.push({ x: -FX, z, sw, w, yaw: 0 });
  }
  // Fixed-direction wheels + encoders on the drive-wheel axis (x = 0), just outside the frame.
  const FW_Z = 0.378;
  // Approximate measuring-wheel dimensions from the reference photo (metres).
  // All three wheel pairs meet the floor on x = 0; radii set their axle heights.
  const ENCODER_R = 0.036, ENCODER_Z = 0.318, ENCODER_BODY_Z = 0.346;
  const ENCODER_TYRE = tyreGeometry(ENCODER_R, 0.018, 0.019);
  function makeEncoderWheel(side) {
    const g = new THREE.Group(), spin = new THREE.Group();
    g.name = side < 0 ? 'Left encoder measuring wheel' : 'Right encoder measuring wheel';
    g.position.set(0, ENCODER_R, side * ENCODER_Z);
    g.add(spin);
    mesh(ENCODER_TYRE, mat.encoderTyre, spin);
    cyl(spin, V(), 0.020, 0.016, 'z', mat.hub);
    for (const face of [-1, 1]) {
      cyl(spin, V(0, 0, face * 0.0085), 0.009, 0.002, 'z', mat.steel);
      for (let i = 0; i < 4; i++) {
        const a = i * Math.PI / 2;
        cyl(spin, V(Math.cos(a) * 0.013, Math.sin(a) * 0.013, face * 0.009), 0.002, 0.002, 'z', mat.cap);
      }
    }
    platform.add(g);
    return { g, spin, angle: 0, r: ENCODER_R };
  }
  const fixed = [];
  for (const side of [-1, 1]) {
    const z = side * FW_Z;
    box(platform, [-0.035, FOOT - 0.006, Math.min(side * (FZ - 0.02), side * (FW_Z + 0.024))], [0.035, FOOT, Math.max(side * (FZ - 0.02), side * (FW_Z + 0.024))], mat.hub);
    for (const dz of [-0.0195, 0.0195]) box(platform, [-0.022, 0.035, z + dz - 0.0025], [0.022, FOOT - 0.006, z + dz + 0.0025], mat.hub);
    const w = makeWheel(mat.fixedRing);
    w.g.position.set(0, 0.05, z);
    platform.add(w.g);
    // The encoder has its own green ground-contact wheel, inboard of the fixed wheel.
    // Its stationary black housing is coaxial with that measuring wheel, not the caster.
    const encoderWheel = makeEncoderWheel(side);
    const enc = new THREE.Group();
    enc.name = side < 0 ? 'Left encoder housing' : 'Right encoder housing';
    enc.position.set(0, ENCODER_R, side * ENCODER_BODY_Z);
    cyl(enc, V(), 0.019, 0.026, 'z', mat.enc);
    cyl(enc, V(0, 0, side * 0.0105), 0.0195, 0.004, 'z', mat.encRing);
    // Shaft bridges the wheel hub and the inner face of the encoder housing.
    cyl(enc, V(0, 0, -side * 0.022), 0.005, 0.022, 'z', mat.steel);
    cyl(enc, V(0, 0, -side * 0.014), 0.008, 0.006, 'z', mat.hub);
    box(enc, [-0.006, 0.019, -0.008], [0.006, 0.026, 0.008], mat.cap);   // cable gland
    platform.add(enc);
    // Narrow rear mounting strap connects the housing to the existing frame bracket.
    box(platform, [-0.025, ENCODER_R, side * ENCODER_BODY_Z - 0.010],
      [-0.019, FOOT - 0.006, side * ENCODER_BODY_Z + 0.010], mat.hub);
    fixed.push({ side, w, enc, encoderWheel });
  }

  /* ------------------------------------------------------------------ models from the bundles */
  const [armB, tracerB] = await bundlesP;
  // Tracer 2.0: drive axle aligned with the frame's mid posts (x = 0)
  const tracer = new THREE.Group();
  platform.add(tracer);
  const tracerZ = new THREE.Group();
  tracerZ.rotation.x = -Math.PI / 2;
  tracerZ.position.set(0.018818, tracerB.man.ground_offset || 0.153, 0);
  tracer.add(tracerZ);
  const T = buildTree(tracerB, tracerZ);
  const tracerMats = [];
  tracer.traverse(o => { if (o.isMesh && !tracerMats.includes(o.material)) tracerMats.push(o.material); });
  // tracer materials are shared only with the tracer; clone so fading the base never touches the arms
  const tmap = new Map(tracerMats.map(m => [m, Object.assign(m.clone(), { userData: { baseOpacity: 1 } })]));
  tracer.traverse(o => { if (o.isMesh) o.material = tmap.get(o.material); });
  const baseMats = [...tmap.values()];

  // Two ARX X5 arms (URDF frame) on the plates, head camera on the mast.
  const armRoot = new THREE.Group();
  armRoot.position.set(ARM_X, ARM_Y, 0);
  armRoot.rotation.x = -Math.PI / 2;
  platform.add(armRoot);
  const HEAD = ['head_camera_mount', 'head_camera_body_approx', 'head_camera_optical_frame'];
  const armL = buildTree(armB, armRoot, { mirrorY: 1 });            // URDF +y = robot's left (three -z)
  const armR = buildTree(armB, armRoot, { mirrorY: -1, skip: HEAD });
  const arms = { [-1]: armL, [1]: armR };                            // keyed by three.js z side

  // Joint poses [q1..q6, finger opening per side (m)], solved offline on the URDF.
  const POSE = {
    rest: [0, 1.15, 1.0, -0.45, 0, 0, 0.02],          // Fig. 4 resting pose
    hold: [0, 0.734, 0.642, -0.169, 0, 0, 0.012],     // operator holds the finger-pad sleeves at chest height
    holdLow: [0, 1.535, 0.599, 0.412, 0, 0, 0.03],
    present: [0, 1.015, 1.085, -0.594, 0, 0, 0.02],
    reach: [0, 2.161, 1.174, -0.322, 0, 0, 0.034],
    lift: [0, 1.383, 1.154, -0.818, 0, 0, 0.006],
  };
  const qCur = { [-1]: POSE.rest.slice(), [1]: POSE.rest.slice() };
  const qDes = { [-1]: POSE.rest.slice(), [1]: POSE.rest.slice() };
  function applyArm(side) {
    const A = arms[side], q = qCur[side];
    for (let i = 0; i < 6; i++) A.set('joint' + (i + 1), q[i]);
    A.set('joint7', q[6]); A.set('joint8', q[6]);
  }
  const mixPose = (out, p, k) => { for (let i = 0; i < 7; i++) out[i] = lerp(out[i], p[i], k); return out; };
  applyArm(-1); applyArm(1);
  // Pad (hand-contact sleeve) centre, in link7 / link8 frames (from the robot description).
  const PAD7 = V(-0.0373, -0.0082, -0.0544), PAD8 = V(-0.0373, 0.0082, -0.0544);
  function padWorld(side, out) {
    const A = arms[side];
    A.links.link7.localToWorld(out.copy(PAD7));
    A.links.link8.localToWorld(_e.copy(PAD8));
    return out.add(_e).multiplyScalar(0.5);
  }

  // Camera frustums from the optical frames (z forward).
  const headFr = makeFrustum(0.42, 86 * DEG, 56 * DEG, 0xf89942);
  armL.links.head_camera_optical_frame.add(headFr);
  const wristFr = [armL, armR].map(A => { const f = makeFrustum(0.24, 86 * DEG, 56 * DEG, 0x2196f3); A.links.wrist_camera_optical_frame.add(f); return f; });
  const frMats = [...headFr.userData.mats, ...wristFr.flatMap(f => f.userData.mats)];

  /* ------------------------------------------------------------------ operator (translucent mannequin) */
  const op = new THREE.Group();
  platform.add(op);
  const opMat = M(0xa9c1e8, { roughness: 0.55, transparent: true, opacity: 0.55 });
  const OPX = 0.16;
  const part = (geo, sx, sy, sz) => { const o = mesh(geo, opMat, op, false); o.scale.set(sx, sy, sz); return o; };
  const opHead = part(SPH, 0.092, 0.112, 0.088); opHead.position.set(OPX + 0.015, 1.635, 0);
  const opNeck = part(CYL, 1, 1, 1); between(opNeck, V(OPX, 1.45, 0), V(OPX + 0.01, 1.56, 0), 0.045);
  const opTorso = part(new THREE.CylinderGeometry(1, 0.82, 1, 28), 0.11, 0.46, 0.175); opTorso.position.set(OPX, 1.2, 0);
  const opPelvis = part(SPH, 0.11, 0.09, 0.165); opPelvis.position.set(OPX, 0.95, 0);
  const limbs = {};
  for (const s of [-1, 1]) {
    limbs[s] = {
      sh: part(SPH, 0.055, 0.055, 0.055), ua: part(CYL, 1, 1, 1), el: part(SPH, 0.04, 0.04, 0.04), fa: part(CYL, 1, 1, 1), hand: part(SPH, 0.045, 0.03, 0.05),
      th: part(CYL, 1, 1, 1), kn: part(SPH, 0.058, 0.058, 0.058), sn: part(CYL, 1, 1, 1), ft: part(BOX, 0.24, 0.06, 0.09),
      E: V(), K: V(), F: V(), H: V(),
    };
  }
  // Relaxed elbows bend down in the shoulder-to-hand plane, never inward
  // across the torso. The robot shoulder yaw supplies the lateral clearance.
  const ELBOW_DOWN = V(0, -1, 0);
  function solveOperatorElbow(shoulder, hand, limb) {
    twoBone(shoulder, hand, 0.295, 0.285, ELBOW_DOWN, limb.E);
  }

  const FWD = V(1, 0, 0.05);
  function updateOperator(phase, amp) {
    for (const s of [-1, 1]) {
      const L = limbs[s];
      const hip = _d.set(OPX, 0.93, 0.09 * s);
      const ph = phase + (s > 0 ? Math.PI : 0);
      L.F.set(OPX + 0.15 * Math.sin(ph) * amp, 0.075 + Math.max(0, Math.cos(ph)) * 0.055 * amp, 0.1 * s);
      twoBone(hip, L.F, 0.44, 0.42, FWD, L.K);
      between(L.th, hip, L.K, 0.068); L.kn.position.copy(L.K);
      between(L.sn, L.K, L.F, 0.052);
      L.ft.position.set(L.F.x + 0.06, 0.03 + Math.max(0, Math.cos(ph)) * 0.05 * amp, L.F.z);
      const shoulder = _e.set(OPX, 1.415, 0.22 * s).clone();
      // Anchor the palm at the midpoint of the two gripper contact pads.
      // Keep this in the gripper frame so yaw and opening cannot offset the hand.
      platform.worldToLocal(padWorld(s, L.H));
      solveOperatorElbow(shoulder, L.H, L);
      L.sh.position.copy(shoulder);
      between(L.ua, shoulder, L.E, 0.046); L.el.position.copy(L.E);
      between(L.fa, L.E, L.H, 0.038); L.hand.position.copy(L.H);
    }
  }

  /* ------------------------------------------------------------------ walking path (mobility and processing) */
  const PA = 1.45, PB = 1.5, PT = 26;
  const P_ROT = Math.atan2(PB, PA);
  function pathXY(s, out) {
    const x = PA * Math.sin(s), y = PB * Math.sin(s) * Math.cos(s);
    const c = Math.cos(-P_ROT), sn = Math.sin(-P_ROT);
    return out.set(x * c - y * sn, x * sn + y * c);
  }
  const _p2a = V2(0, 0), _p2b = V2(0, 0);
  function pathPose(s, outPos) {
    pathXY(s, _p2a); pathXY(s + 1e-3, _p2b);
    outPos.set(_p2a.x, 0, -_p2a.y);
    return Math.atan2(_p2b.y - _p2a.y, _p2b.x - _p2a.x);
  }
  function pathVel(s) {
    const k = 2 * Math.PI / PT, ds = 2e-3;
    pathXY(s - ds, _p2a); pathXY(s + ds, _p2b);
    const v = _p2a.distanceTo(_p2b) / (2 * ds) * k;
    const y0 = pathPose(s - ds, _a), y1 = pathPose(s + ds, _a);
    return { v, w: wrap(y1 - y0) / (2 * ds) * k };
  }
  const pathPts = [], bb = new THREE.Box3();
  for (let i = 0; i <= 256; i++) { pathXY(i / 256 * Math.PI * 2, _p2a); const p = V(_p2a.x, 0.004, -_p2a.y); pathPts.push(p); bb.expandByPoint(p); }
  const pathCenter = bb.getCenter(V()), pathHalf = bb.getSize(V()).multiplyScalar(0.5);

  const kin = new THREE.Group(); scene.add(kin);
  const pathLine = dashedLine(pathPts, 0x767171, 0.07, 0.05, kin);
  const axle = dashedLine([V(0, 0.012, -6), V(0, 0.012, 6)], 0x2196f3, 0.05, 0.04, platform);
  const heading = new THREE.ArrowHelper(V(1, 0, 0), V(0, 0.03, 0), 0.6, 0x70ad47, 0.09, 0.06);
  platform.add(heading);
  const TRACK = 2 * FW_Z;
  const wheelArrows = [-1, 1].map(s => { const a = new THREE.ArrowHelper(V(1, 0, 0), V(0, 0.03, s * FW_Z), 0.3, 0x00b0f0, 0.07, 0.05); platform.add(a); return a; });
  const kinMats = [pathLine.material, axle.material, heading.line.material, heading.cone.material, ...wheelArrows.flatMap(a => [a.line.material, a.cone.material])];
  kinMats.forEach(m => { m.transparent = true; m.depthWrite = false; });

  /* ------------------------------------------------------------------ labels */
  const pl = (o, x, y, z) => platform.localToWorld(o.set(x, y, z));
  const wp = (obj, o, x = 0, y = 0, z = 0) => obj.localToWorld(o.set(x, y, z));
  const GREY = '#8a929c';
  const LBL = [
    { s: [0], key: 1, text: '6-DoF arm', dot: GREY, side: 'left', get: o => wp(armR.links.link3, o, 0.12, 0, -0.03) },
    { s: [0], key: 0, text: 'Wrist camera', dot: '#2196F3', side: 'left', get: o => wp(armR.links.wrist_camera_body_approx, o) },
    { s: [0], key: 1, text: 'Head camera', dot: '#F89942', side: 'right', get: o => wp(armL.links.head_camera_body_approx, o) },
    { s: [0], key: 0, text: 'Framework · 880 × 710 mm', dot: GREY, side: 'right', get: o => pl(o, -0.25, TOPC, FZ) },
    { s: [0], key: 1, text: 'Differential-drive base', dot: '#50555c', side: 'left', get: o => wp(tracerZ, o, 0.3, -0.1, 0.0) },
    { s: [0], key: 0, text: 'Drive wheel', dot: '#92D050', side: 'left', get: o => wp(T.links.right_link, o) },
    { s: [0], key: 0, text: 'Encoder', dot: '#9D68F4', side: 'right', get: o => wp(fixed[1].enc, o) },
    { s: [0], key: 1, text: 'Fixed-direction wheel', dot: '#00B0F0', side: 'right', get: o => wp(fixed[1].w.g, o, 0, 0, 0.016) },
    { s: [0], key: 0, text: 'Passive wheel', dot: GREY, side: 'right', get: o => wp(casters[1].w.g, o) },

    { s: [1], key: 1, text: 'Natural locomotion', dot: '#A9C1E8', side: 'left', get: o => pl(o, OPX, 0.5, 0.1) },
    { s: [1], key: 1, text: 'Kinesthetic teaching', dot: '#d6452c', side: 'left', get: o => padWorld(1, o) },
    { s: [1], key: 0, text: 'Passive wheel', dot: GREY, side: 'right', get: o => wp(casters[1].w.g, o) },
    { s: [1], key: 1, text: 'Encoder', dot: '#9D68F4', side: 'right', get: o => wp(fixed[1].enc, o) },
    { s: [1], key: 1, text: 'Fixed-direction wheel', dot: '#00B0F0', side: 'right', get: o => wp(fixed[1].w.g, o, 0, 0, 0.016) },

    { s: [2], key: 1, tag: 1, text: 'drive-wheel axis', dot: '#2196F3', get: o => pl(o, 0, 0.012, 1.15) },
    { s: [2], key: 1, tag: 1, text: 'Nonholonomic motion', dot: '#70AD47', get: o => pl(o, 0.8, 0.03, 0) },
    { s: [2], key: 0, tag: 1, text: 'u_L', dot: '#00B0F0', get: o => pl(o, 0.1, 0.03, -FW_Z - 0.1) },
    { s: [2], key: 0, tag: 1, text: 'u_R', dot: '#00B0F0', get: o => pl(o, 0.1, 0.03, FW_Z + 0.1) },

    // Anchor lies on triangle 437 of long_finger_left.STL, in link7 coordinates.
    { s: [3], key: 1, text: 'End-effector', dot: '#d6452c', side: 'right', get: o => wp(armR.links.link7, o, 0.04789276097759467, -0.014416219017827622, -0.043023064006510514) },
    { s: [3], key: 1, text: 'Hand contact area', dot: '#e3e1d6', side: 'right', get: o => wp(armR.links.link8, o, -0.0373, 0.0082, -0.0544) },
    { s: [3], key: 1, text: 'Wrist camera', dot: '#2196F3', side: 'left', get: o => wp(armR.links.wrist_camera_body_approx, o) },

    { s: [4], key: 1, text: 'Head camera image', dot: '#F89942', side: 'right', get: o => wp(armL.links.head_camera_optical_frame, o, 0, 0, 0.2) },
    { s: [4], key: 1, text: 'Wrist camera images', dot: '#2196F3', side: 'left', get: o => wp(armL.links.wrist_camera_optical_frame, o, 0, 0, 0.12) },
    { s: [4], key: 0, text: 'Arm actions', dot: GREY, side: 'left', get: o => wp(armL.links.link3, o, 0.12, 0, -0.03) },
    { s: [4], key: 1, text: 'Base velocities v, ω', dot: '#92D050', side: 'right', get: o => wp(tracerZ, o, 0.0, 0.3, 0.02) },
  ];
  const SVGNS = 'http://www.w3.org/2000/svg';
  for (const d of LBL) {
    const el = document.createElement('div');
    el.className = 'mech-label' + (d.light ? ' mech-label--light' : '');
    const i = document.createElement('i'); i.style.background = d.dot;
    const span = document.createElement('span');
    span.textContent = typeof d.text === 'function' ? d.text() : d.text;
    el.append(i, span);
    labelLayer.appendChild(el);
    const line = document.createElementNS(SVGNS, 'line');
    const circ = document.createElementNS(SVGNS, 'circle');
    circ.setAttribute('r', '3.2');
    svg.append(line, circ);
    Object.assign(d, { el, span, line, circ, w: 0, on: false, pos: V(), lastText: '' });
  }
  function measureLabels() { for (const d of LBL) d.w = d.el.offsetWidth; }
  const placed = { left: [], right: [] };
  function layoutLabels(ready) {
    const w = host.clientWidth, h = host.clientHeight;
    const callout = w >= 640;
    placed.left.length = placed.right.length = 0;
    const tags = [];
    for (const d of LBL) {
      let on = ready && d.s.includes(active) && (callout || d.key) && (!d.when || d.when());
      if (on) {
        d.get(d.pos);
        _a.copy(d.pos).project(camera);
        if (_a.z > 1 || Math.abs(_a.x) > 1.05 || Math.abs(_a.y) > 1.05) on = false;
        d.ax = (_a.x * 0.5 + 0.5) * w; d.ay = (-_a.y * 0.5 + 0.5) * h;
        if (typeof d.text === 'function') {
          const t = d.text();
          if (t !== d.lastText) { const grow = t.length !== d.lastText.length; d.span.textContent = t; d.lastText = t; if (grow) d.w = d.el.offsetWidth; }
        }
      }
      if (on !== d.on) {
        d.on = on;
        d.el.classList.toggle('is-on', on);
        d.line.style.opacity = d.circ.style.opacity = on ? 1 : 0;
      }
      if (!on) continue;
      if (callout && !d.tag) placed[d.side || (d.ax < w / 2 ? 'left' : 'right')].push(d);
      else tags.push(d);
    }
    tags.sort((p, q) => p.ay - q.ay);
    for (let i = 0; i < tags.length; i++) {
      const d = tags[i], lw = d.w || 80;
      d.tx = clamp(d.ax + 10, 6, w - lw - 6);
      d.ty = clamp(d.ay - 12, 46, h - 32);
      for (let guard = 0; guard < 8; guard++) {
        let hit = null;
        for (let j = 0; j < i; j++) {
          const o = tags[j];
          if (d.tx < o.tx + (o.w || 80) + 4 && o.tx < d.tx + lw + 4 && d.ty < o.ty + 27 && o.ty < d.ty + 27) { hit = o; break; }
        }
        if (!hit) break;
        d.ty = hit.ty + 28;
      }
      d.ty = Math.min(d.ty, h - 30);
      d.el.style.transform = `translate(${Math.round(d.tx)}px, ${Math.round(d.ty)}px)`;
      d.circ.setAttribute('cx', d.ax.toFixed(1)); d.circ.setAttribute('cy', d.ay.toFixed(1));
      const far = Math.abs(d.ty + 12 - d.ay) > 8 || d.tx > d.ax + 14 || d.tx + lw < d.ax;
      const lx = d.tx > d.ax ? d.tx : d.tx + lw;
      d.line.setAttribute('x1', d.ax.toFixed(1)); d.line.setAttribute('y1', d.ay.toFixed(1));
      d.line.setAttribute('x2', (far ? lx : d.ax).toFixed(1)); d.line.setAttribute('y2', (far ? d.ty + 12 : d.ay).toFixed(1));
    }
    const GAP = 30, PAD = 22;
    for (const side of ['left', 'right']) {
      const list = placed[side].sort((p, q) => p.ay - q.ay);
      let prev = -Infinity;
      for (const d of list) { d.ly = Math.max(clamp(d.ay, PAD + 40, h - PAD), prev + GAP); prev = d.ly; }
      let next = Infinity;
      for (let i = list.length - 1; i >= 0; i--) { const d = list[i]; d.ly = Math.min(d.ly, Math.min(h - PAD, next - GAP)); next = d.ly; }
      for (const d of list) {
        const x = side === 'left' ? 14 : w - 14 - d.w;
        d.el.style.transform = `translate(${Math.round(x)}px, ${Math.round(d.ly - 12)}px)`;
        const lx = side === 'left' ? x + d.w + 4 : x - 4;
        d.line.setAttribute('x1', d.ax.toFixed(1)); d.line.setAttribute('y1', d.ay.toFixed(1));
        d.line.setAttribute('x2', lx.toFixed(1)); d.line.setAttribute('y2', d.ly.toFixed(1));
        d.circ.setAttribute('cx', d.ax.toFixed(1)); d.circ.setAttribute('cy', d.ay.toFixed(1));
      }
    }
  }

  /* ------------------------------------------------------------------ controls & camera rig */
  let controls = null, userOrbit = false;
  if (mqFine.matches) {
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableZoom = false;
    controls.enablePan = false;
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.7;
    controls.minPolarAngle = 0;
    controls.maxPolarAngle = 1.5;
    controls.addEventListener('start', () => { userOrbit = true; stage.classList.add('is-touched'); wake(); });
    controls.addEventListener('change', () => wake(2));
  } else if (hint) hint.hidden = true;

  const rig = { target: V(), offset: V() };
  const manipulationFocus = V();
  let tween = null, snap = true;
  const _pt = V(), _po = V(), sphA = new THREE.Spherical(), sphB = new THREE.Spherical();
  function fitScale() { const asp = camera.aspect; return asp >= 1.3 ? 1 : Math.pow(1.3 / asp, 0.9); }
  function preset(i, Tt, O) {
    const p = platform.position;
    switch (i) {
      case 0: Tt.set(0.06, 0.64, 0); O.set(2.2, 0.85, 1.45); break;
      case 1: Tt.set(p.x + 0.12, 0.74, p.z); O.set(2.55, 1.0, 1.85); break;
      case 2: {
        const t = Math.tan((camera.fov * DEG) / 2);
        const H = 1.15 * Math.max((pathHalf.z + 1.1) / t, (pathHalf.x + 1.1) / (t * camera.aspect));   // arms reach ahead of the axle and sit 1.2 m nearer the camera
        Tt.copy(pathCenter); Tt.y = 0; O.set(0, H, 0.001); return;
      }
      case 3:
        // Hold the framing steady while both hands move; follow only the platform.
        platform.localToWorld(Tt.copy(manipulationFocus));
        O.set(0.85, 0.30, 1.05).applyAxisAngle(UP, platform.rotation.y);
        break;
      default: Tt.set(p.x + 0.12, 0.6, p.z); O.set(1.75, 0.95, -2.55);
    }
    O.multiplyScalar(fitScale());
  }
  function updateCamera(dt) {
    preset(active, _pt, _po);
    if (snap || reduce) {
      if (snap || !userOrbit) { rig.target.copy(_pt); rig.offset.copy(_po); }
      snap = false; tween = null;
    }
    if (tween) {
      if (controls) controls.update();
      if (tween.delay > 0) { tween.delay = Math.max(0, tween.delay - dt); return; }
      tween.t = Math.min(1, tween.t + dt / tween.dur);
      const k = ease(tween.t);
      rig.target.lerpVectors(tween.from, _pt, k);
      sphB.setFromVector3(_po);
      sphA.radius = lerp(tween.sph.radius, sphB.radius, k);
      sphA.phi = lerp(tween.sph.phi, sphB.phi, k);
      sphA.theta = tween.sph.theta + wrap(sphB.theta - tween.sph.theta) * k;
      rig.offset.setFromSpherical(sphA);
      camera.position.copy(rig.target).add(rig.offset);
      camera.lookAt(rig.target);
      if (controls) controls.target.copy(rig.target);
      if (tween.t >= 1) tween = null;
      return;
    }
    if (controls && userOrbit) rig.offset.subVectors(camera.position, controls.target);
    rig.target.copy(_pt);
    if (active === 0 && !reduce && !userOrbit) rig.offset.applyAxisAngle(UP, dt * 0.11);
    camera.position.copy(rig.target).add(rig.offset);
    if (controls) { controls.target.copy(rig.target); controls.update(); }
    else camera.lookAt(rig.target);
  }
  function startTween() {
    if (reduce) { snap = true; return; }
    tween = { t: 0, delay: active === 3 ? 0.65 : 0, dur: active === 3 ? 1.7 : 1.5, from: rig.target.clone(), sph: new THREE.Spherical().setFromVector3(rig.offset) };
  }

  /* ------------------------------------------------------------------ state + update */
  const W = { baseOut: 0, op: 0, move: 0, pathMix: 0, kin: 0, deploy: 0 };
  function approach(k, target, rate, dt) {
    if (reduce) { W[k] = target; return; }
    W[k] += (target - W[k]) * (1 - Math.exp(-rate * dt));
    if (Math.abs(W[k] - target) < 2e-3) W[k] = target;
  }
  // Display order: overview, mobility, manipulation, processing, policy.
  // Scene IDs stay stable so camera presets, overlays and callouts remain paired.
  const SCENE_ORDER = [0, 1, 3, 2, 4];
  let active = 0, stepClock = 0, armClock = 0, pathS = 0, walkPhase = 0, readoutTimer = 0;
  const prevPos = V(), vel = { v: 0, w: 0 }, _pp = V();
  let prevYaw = 0;
  const SEQ_T = 10;
  const tmpPose = [0, 0, 0, 0, 0, 0, 0];
  // Deployment demo: reach, grasp, lift, drive back and forth, place, rest.
  function deployPose(t, side, out) {
    const key = [[0, 'rest'], [1.5, 'reach'], [2.2, 'reachC'], [3.4, 'lift'], [7.0, 'lift'], [8.0, 'reachC'], [8.5, 'reach'], [10, 'rest']];
    const P = n => n === 'reachC' ? [...POSE.reach.slice(0, 6), 0.006] : POSE[n];
    let i = 0;
    while (i < key.length - 2 && t >= key[i + 1][0]) i++;
    const [t0, n0] = key[i], [t1, n1] = key[i + 1];
    const k = ease(clamp((t - t0) / (t1 - t0), 0, 1));
    const A = P(n0), B = P(n1);
    for (let j = 0; j < 7; j++) out[j] = lerp(A[j], B[j], k);
    out[0] += side * 0.12 * Math.sin(Math.PI * clamp(t / 8.5, 0, 1));   // slight inward yaw while working
    return out;
  }
  function deployDx(t) {
    if (t < 3.4) return 0;
    if (t < 5.2) return -0.35 * ease((t - 3.4) / 1.8);
    if (t < 7.0) return -0.35 * (1 - ease((t - 5.2) / 1.8));
    return 0;
  }

  function update(dt) {
    const tdt = reduce ? 0 : dt;
    stepClock += tdt;
    armClock += tdt;
    const s = active, collecting = s === 1 || s === 2;
    approach('baseOut', collecting || s === 3 ? 1 : (W.op < 0.3 && W.move < 0.25 ? 0 : 1), 3.2, dt);
    const opT = SHOW_OPERATOR && (collecting || s === 3) && W.baseOut > 0.9 ? 1 : 0;
    approach('op', opT, opT ? 4 : 6, dt);
    const mvT = collecting && (W.op > 0.88 || !SHOW_OPERATOR) && W.baseOut > 0.9 ? 1 : 0;
    approach('move', mvT, mvT ? 1.6 : (s === 3 ? 5.5 : 3.5), dt);
    // Speed can reach zero without pulling the parked platform back to the origin.
    if (s === 3 && stepClock >= 0.65) W.move = 0;
    if (s !== 3) approach('pathMix', collecting ? 1 : 0, collecting ? 1.6 : 3.5, dt);
    approach('kin', s === 2 ? 1 : 0, 3, dt);
    approach('deploy', s === 4 && W.baseOut < 0.3 ? 1 : 0, 2.6, dt);

    // Tracer slides out through the open front of the frame
    tracer.position.x = ease(W.baseOut) * 1.3;
    const bo = 1 - smooth(0.5, 0.95, W.baseOut);
    tracer.visible = bo > 0.005;
    setOpacity(baseMats, bo);

    // platform pose
    if (reduce && s !== 3) pathS = collecting ? 1.05 : 0;
    else if (W.move > 0) pathS += tdt * (2 * Math.PI / PT) * W.move;
    else if (!collecting && s !== 3 && W.pathMix === 0) pathS = 0;
    const yawPath = pathPose(pathS, _pp);
    const tSeq = reduce ? 3.6 : stepClock % SEQ_T;
    const dx = deployDx(tSeq) * W.deploy;
    platform.position.set(lerp(dx, _pp.x, W.pathMix), 0, lerp(0, _pp.z, W.pathMix));
    platform.rotation.y = lerpAngle(0, yawPath, W.pathMix);
    platform.updateMatrixWorld();

    // body velocity -> wheels, casters, encoders
    const yaw = platform.rotation.y;
    if (dt > 0 && tdt > 0) {
      _a.subVectors(platform.position, prevPos);
      const v = _a.x * Math.cos(yaw) - _a.z * Math.sin(yaw);
      vel.v = lerp(vel.v, v / dt, 0.5);
      vel.w = lerp(vel.w, wrap(yaw - prevYaw) / dt, 0.5);
    } else { vel.v = 0; vel.w = 0; }
    prevPos.copy(platform.position); prevYaw = yaw;
    const uL = vel.v - vel.w * TRACK / 2, uR = vel.v + vel.w * TRACK / 2;
    for (const f of fixed) {
      const u = f.side < 0 ? uL : uR;
      f.w.angle += u * tdt / f.w.r;
      f.w.spin.rotation.z = -f.w.angle;
      // The measuring wheel rolls independently at its own lateral position and radius.
      const encoderSpeed = vel.v + vel.w * f.encoderWheel.g.position.z;
      f.encoderWheel.angle += encoderSpeed * tdt / f.encoderWheel.r;
      f.encoderWheel.spin.rotation.z = -f.encoderWheel.angle;
    }
    for (const c of casters) {
      const vx = vel.v + vel.w * c.z, vz = -vel.w * c.x, sp = Math.hypot(vx, vz);
      if (sp > 0.015) c.yaw = lerpAngle(c.yaw, Math.atan2(-vz, vx), 1 - Math.exp(-10 * tdt));
      c.sw.rotation.y = c.yaw;
      c.w.angle += sp * tdt / c.w.r;
      c.w.spin.rotation.z = -c.w.angle;
    }
    if (tracer.visible) { // drive wheels: left joint axis points right, right joint axis points left
      const dl = (vel.v - vel.w * 0.2537) * tdt / 0.0625, dr = (vel.v + vel.w * 0.2537) * tdt / 0.0625;
      T.joints.left_joint.value -= dl; T.joints.right_joint.value += dr;
      const jl = T.joints.left_joint, jr = T.joints.right_joint;
      jl.motion.quaternion.setFromAxisAngle(jl.axis, jl.value);
      jr.motion.quaternion.setFromAxisAngle(jr.axis, jr.value);
    }

    // arms
    const t = armClock;
    for (const sd of [-1, 1]) {
      const D = qDes[sd];
      for (let i = 0; i < 7; i++) D[i] = POSE.rest[i];
      if (W.op > 0) {
        const k = 0.5 + 0.5 * Math.sin(0.9 * t + (sd > 0 ? 0.8 : 0));
        for (let i = 0; i < 7; i++) tmpPose[i] = lerp(POSE.hold[i], POSE.holdLow[i], k * 0.55);
        // A slight mirrored shoulder yaw opens the robot elbows outward,
        // leaving the interior space for the operator's elbows.
        tmpPose[0] = sd * (0.18 + 0.025 * Math.sin(0.6 * t));
        mixPose(D, tmpPose, W.op);
      }
      if (W.deploy > 0) mixPose(D, deployPose(tSeq, sd, tmpPose), W.deploy);
      const k = reduce ? 1 : 1 - Math.exp(-(W.deploy > 0.5 ? 9 : 6) * dt);
      mixPose(qCur[sd], D, k);
      applyArm(sd);
    }
    platform.updateMatrixWorld(true);

    // operator
    op.visible = W.op > 0.01;
    if (op.visible) {
      opMat.opacity = 0.55 * W.op;
      walkPhase += Math.abs(vel.v) * tdt / 0.5 * Math.PI;
      updateOperator(walkPhase, clamp(Math.abs(vel.v) / 0.3, 0, 1));
    }

    setOpacity(frMats, W.deploy);
    headFr.visible = W.deploy > 0.01;
    wristFr.forEach(f => (f.visible = headFr.visible));

    // kinematics overlay
    const kv = W.kin;
    kin.visible = axle.visible = heading.visible = kv > 0.01;
    wheelArrows.forEach(a => (a.visible = kv > 0.01));
    if (kv > 0.01) {
      kinMats.forEach(m => (m.opacity = kv));
      const pv = reduce ? pathVel(pathS) : vel;
      heading.setLength(0.45 + Math.max(0, pv.v) * 1.2, 0.09, 0.06);
      const ul = pv.v - pv.w * TRACK / 2, ur = pv.v + pv.w * TRACK / 2;
      [ul, ur].forEach((u, i) => {
        const ar = wheelArrows[i];
        ar.setDirection(_a.set(Math.sign(u) || 1, 0, 0));
        ar.setLength(Math.max(0.08, Math.abs(u) * 1.4), 0.07, 0.05);
      });
      readoutTimer -= dt;
      if (readout && (readoutTimer <= 0 || reduce)) {
        readoutTimer = 0.12;
        readout.querySelector('[data-r="ul"]').textContent = ul.toFixed(2);
        readout.querySelector('[data-r="ur"]').textContent = ur.toFixed(2);
        readout.querySelector('[data-r="v"]').textContent = pv.v.toFixed(2);
        readout.querySelector('[data-r="w"]').textContent = pv.w.toFixed(2);
      }
    }
    if (readout) readout.hidden = !(s === 2 && kv > 0.5);

    sun.target.position.set(rig.target.x, 0, rig.target.z);
    sun.position.copy(sun.target.position).add(SUN_OFF);
  }

  /* ------------------------------------------------------------------ step switching */
  function setStep(i) {
    const sceneId = SCENE_ORDER[i];
    if (sceneId === active && stage.dataset.step === String(i)) return;
    if (sceneId === 3) {
      // Capture the hands' midpoint once; do not make the camera chase their oscillation.
      padWorld(-1, manipulationFocus);
      padWorld(1, _pt);
      manipulationFocus.add(_pt).multiplyScalar(0.5);
      platform.worldToLocal(manipulationFocus);
    }
    active = sceneId;
    stage.dataset.step = String(i);
    steps.forEach((el, k) => el.classList.toggle('is-active', k === i));
    dots.forEach((d, k) => d.setAttribute('aria-current', k === i ? 'step' : 'false'));
    if (captionEl) captionEl.textContent = steps[i].getAttribute('data-title') || '';
    stepClock = 0;
    userOrbit = false;
    stage.classList.remove('is-touched');
    startTween();
    wake(90);
  }
  function marker() { return window.innerHeight * (mqDesktop.matches ? 0.5 : 0.8); }
  function onScroll() {
    const m = marker();
    let idx = 0;
    steps.forEach((el, k) => { if (el.getBoundingClientRect().top <= m) idx = k; });
    setStep(idx);
  }
  let scrollTick = false;
  window.addEventListener('scroll', () => {
    if (scrollTick) return;
    scrollTick = true;
    requestAnimationFrame(() => { scrollTick = false; onScroll(); });
  }, { passive: true });
  function goTo(k) {
    const el = steps[k];
    const y = el.getBoundingClientRect().top + window.pageYOffset - marker() + 24;
    window.scrollTo({ top: Math.max(0, y), behavior: reduce ? 'auto' : 'smooth' });
    setStep(k);
  }
  dots.forEach((d, k) => d.addEventListener('click', () => goTo(k)));
  steps.forEach((el, k) => el.addEventListener('click', ev => { if (!ev.target.closest('a')) goTo(k); }));

  /* ------------------------------------------------------------------ loop / visibility / resize */
  let framesLeft = Infinity, running = false, last = performance.now();
  function wake(n = 6) { framesLeft = reduce ? Math.max(framesLeft === Infinity ? 0 : framesLeft, n) : Infinity; }
  function frame(now) {
    // window.__haloFixedDt (seconds) forces a fixed timestep; only used by automated screenshot checks.
    const dt = window.__haloFixedDt || Math.min(0.05, Math.max(0, (now - last) / 1000));
    last = now;
    if (reduce && framesLeft <= 0) return;
    if (reduce) framesLeft--;
    update(dt);
    updateCamera(dt);
    renderer.render(scene, camera);
    layoutLabels(!tween || tween.t > 0.86);
  }
  let inView = false;
  function setRunning() {
    const want = inView && !document.hidden;
    if (want === running) return;
    running = want;
    last = performance.now();
    renderer.setAnimationLoop(want ? frame : null);
  }
  new IntersectionObserver(es => { inView = es[0].isIntersecting; setRunning(); }, { rootMargin: '200px 0px' }).observe(stage);
  document.addEventListener('visibilitychange', setRunning);
  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    measureLabels();
    if (!userOrbit && !tween) snap = true;
    wake(4);
  }
  new ResizeObserver(resize).observe(host);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { measureLabels(); wake(2); });
  if (mqReduce.addEventListener) mqReduce.addEventListener('change', () => { reduce = mqReduce.matches; wake(30); });

  root.classList.remove('is-loading');
  root.classList.add('is-live');
  resize();
  active = -1;
  setStep(0);
  snap = true;
  onScroll();
  root.haloViewer = { setStep, get step() { return active; }, renderer };
}
