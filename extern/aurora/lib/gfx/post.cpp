#include "post.hpp"

#include "../logging.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "recording.hpp"

#include <aurora/gfx.hpp>

#include <array>
#include <cstring>
#include <string>

// Full-screen atmosphere, clouds and finish passes (see post.hpp).
namespace aurora::gfx::post {
namespace {
Module Log("aurora::gfx::post");
using webgpu::g_device;

constexpr const char* Source = R"(
struct Params {
  frustum: vec4f,
  depth: vec4f,
  view: mat3x4f,
  prevView: mat3x4f,
  fog: vec4f,
  fogColor: vec4f,
  sun: vec4f,
  sunColor: vec4f,
  lights: array<vec4f, 16>,
  water: vec4f,
  waterBox: vec4f,
  weather: vec4f,
  clouds: vec4f,
  cloudColor: vec4f,
  finish: vec4f,
  misc: vec4u,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var depthTex: DEPTH_TYPE;
@group(0) @binding(2) var src: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VertexOutput;
  let c = corners[i];
  out.pos = vec4f(c, 0.0, 1.0);
  out.uv = vec2f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
  return out;
}

// The raw depth's z in GX's range: >= 0.99999 nothing drawn (the sky), < depth.z the viewmodel.
fn raw_z(at: vec2i) -> f32 {
  let size = vec2i(textureDimensions(depthTex));
  return 1.0 - textureLoad(depthTex, clamp(at, vec2i(0), size - vec2i(1)), 0);
}

fn is_sky(z: f32) -> bool {
  return z >= 0.99999;
}

fn is_viewmodel(z: f32) -> bool {
  return z < p.depth.z;
}

fn linear_of(z: f32) -> f32 {
  let d = clamp((z - p.depth.z) / max(p.depth.w - p.depth.z, 1e-6), 0.0, 1.0);
  let near = p.depth.x;
  let far = p.depth.y;
  return near * far / (far - d * (far - near));
}

fn view_ray(uv: vec2f) -> vec3f {
  return vec3f(mix(p.frustum.x, p.frustum.y, uv.x), mix(p.frustum.z, p.frustum.w, 1.0 - uv.y), -1.0);
}

fn to_world(v: vec3f) -> vec3f {
  return vec4f(v, 1.0) * p.view;
}

fn to_world_dir(v: vec3f) -> vec3f {
  return vec4f(v, 0.0) * p.view;
}

fn eye() -> vec3f {
  return vec3f(p.view[0].w, p.view[1].w, p.view[2].w);
}

fn srgb_enc(c: vec3f) -> vec3f {
  let l = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(l, vec3f(1.0 / 2.4)) - 0.055, 12.92 * l, l <= vec3f(0.0031308));
}

fn srgb_dec(c: vec3f) -> vec3f {
  let e = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(pow((e + 0.055) / 1.055, vec3f(2.4)), e / 12.92, e <= vec3f(0.04045));
}

fn ign(px: vec2f) -> f32 {
  let f = f32(p.misc.z % 64u) * 5.588238;
  return fract(52.9829189 * fract(dot(px + f, vec2f(0.06711056, 0.00583715))));
}

fn hash3(q: vec3f) -> f32 {
  var r = fract(q * vec3f(0.1031, 0.1030, 0.0973));
  r += dot(r, r.yxz + 33.33);
  return fract((r.x + r.y) * r.z);
}

fn noise3(x: vec3f) -> f32 {
  let i = floor(x);
  let f = fract(x);
  let u = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(hash3(i), hash3(i + vec3f(1.0, 0.0, 0.0)), u.x),
              mix(hash3(i + vec3f(0.0, 1.0, 0.0)), hash3(i + vec3f(1.0, 1.0, 0.0)), u.x), u.y);
  let b = mix(mix(hash3(i + vec3f(0.0, 0.0, 1.0)), hash3(i + vec3f(1.0, 0.0, 1.0)), u.x),
              mix(hash3(i + vec3f(0.0, 1.0, 1.0)), hash3(i + vec3f(1.0, 1.0, 1.0)), u.x), u.y);
  return mix(a, b, u.z);
}

fn fbm(x: vec3f, octaves: u32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = x;
  for (var i = 0u; i < octaves; i++) {
    s += a * noise3(q);
    q = q * 2.03 + vec3f(1.7, -3.1, 0.6);
    a *= 0.5;
  }
  return s;
}

fn hg(cosT: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (12.566371 * pow(max(1.0 + g2 - 2.0 * g * cosT, 1e-4), 1.5));
}

// The world normal from the depth around a pixel (the nearer neighbour on each axis).
fn world_normal(at: vec2i, uv: vec2f, zl: f32) -> vec3f {
  let size = vec2f(textureDimensions(depthTex));
  let texel = 1.0 / size;
  let pos = view_ray(uv) * zl;
  var dx = vec3f(0.0);
  var dy = vec3f(0.0);
  let zr = raw_z(at + vec2i(1, 0));
  let zlf = raw_z(at - vec2i(1, 0));
  let zd = raw_z(at + vec2i(0, 1));
  let zu = raw_z(at - vec2i(0, 1));
  let lr = linear_of(zr);
  let ll = linear_of(zlf);
  let ld = linear_of(zd);
  let lu = linear_of(zu);
  if (abs(lr - zl) < abs(ll - zl)) {
    dx = view_ray(uv + vec2f(texel.x, 0.0)) * lr - pos;
  } else {
    dx = pos - view_ray(uv - vec2f(texel.x, 0.0)) * ll;
  }
  if (abs(ld - zl) < abs(lu - zl)) {
    dy = view_ray(uv + vec2f(0.0, texel.y)) * ld - pos;
  } else {
    dy = pos - view_ray(uv - vec2f(0.0, texel.y)) * lu;
  }
  var n = normalize(cross(dy, dx) + vec3f(0.0, 0.0, 1e-6));
  if (dot(n, -pos) < 0.0) {
    n = -n;
  }
  return normalize(to_world_dir(n));
}

fn caustic(q: vec2f, t: f32) -> f32 {
  var c = 0.0;
  var uv = q * 0.55;
  for (var i = 0; i < 3; i++) {
    let fi = f32(i);
    uv = uv + vec2f(sin(uv.y * 1.37 + t * (0.55 + 0.1 * fi) + fi * 1.7), cos(uv.x * 1.21 - t * (0.47 + 0.12 * fi) - fi)) * 0.45;
    let w = abs(sin(uv.x * 2.3 + t * 0.3) + sin(uv.y * 2.1 - t * 0.35) + sin((uv.x + uv.y) * 1.6 + t * 0.25));
    c += pow(clamp(1.0 - w * 0.55, 0.0, 1.0), 5.0);
  }
  return c;
}

fn fog_density(w: vec3f, t: f32) -> f32 {
  let h = exp(-max(w.z - p.fog.z, 0.0) * p.fog.y);
  let n = fbm(w * 0.18 + vec3f(t * 0.25, t * 0.11, t * 0.05), 3u);
  return p.fog.x * h * (1.0 - p.fog.w + p.fog.w * n * 2.0);
}

@fragment
fn fs_atmosphere(in: VertexOutput) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(src));
  let px = min(vec2i(floor(in.pos.xy)), size - vec2i(1));
  let f = textureLoad(src, px, 0);
  let dsize = vec2i(textureDimensions(depthTex));
  let dat = min(vec2i(in.uv * vec2f(dsize)), dsize - vec2i(1));
  let z = raw_z(dat);
  if (is_viewmodel(z)) {
    return f;
  }
  let t = p.weather.y;
  var c = srgb_dec(f.rgb);
  let ray = view_ray(in.uv);
  let maxDist = p.sunColor.w;
  var zl = maxDist;
  if (!is_sky(z)) {
    zl = linear_of(z);
  }
  let viewPos = ray * zl;
  let world = to_world(viewPos);
  let e = eye();

  if (!is_sky(z)) {
    let n = world_normal(dat, in.uv, zl);
    // Caustics: under the water's surface (in its box), or anywhere when the eye is under it.
    if (p.water.y > 0.0) {
      let below = p.water.x - world.z;
      let inBox = all(world.xy >= p.waterBox.xy) && all(world.xy <= p.waterBox.zw);
      if ((p.water.z > 0.5 || inBox) && below > 0.0 && below < p.water.w) {
        let fade = (1.0 - below / p.water.w) * smoothstep(0.0, 0.25, below);
        let facing = clamp(n.z * 0.8 + 0.3, 0.0, 1.0);
        let k = caustic(world.xy + n.xy * below * 0.3, t) * p.water.y * fade * facing;
        c = c * (1.0 + k * 1.6) + vec3f(0.35, 0.55, 0.6) * k * 0.05;
      }
    }
    // Snow settled on what faces up, in patches.
    if (p.weather.z > 0.0 && n.z > 0.45) {
      let patchy = smoothstep(0.35, 0.6, fbm(world * 0.7, 3u) + (n.z - 0.7) * 0.6);
      let cover = p.weather.z * patchy * smoothstep(0.45, 0.8, n.z);
      let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
      let snow = vec3f(0.86, 0.9, 0.97) * clamp(lum * 2.4 + 0.08, 0.12, 1.0);
      c = mix(c, snow, cover * 0.85);
    }
    // Wet: what faces the rain darkens, gains a sheen and ripples.
    if (p.weather.x > 0.0 && n.z > 0.55) {
      let w = p.weather.x * smoothstep(0.55, 0.85, n.z) * (1.0 - smoothstep(30.0, 60.0, zl));
      c = c * (1.0 - 0.3 * w);
      let cell = floor(world.xy * 2.0);
      let local = fract(world.xy * 2.0) - 0.5;
      let ph = hash3(vec3f(cell, 7.0));
      let age = fract(t * 1.3 + ph);
      let ring = exp(-pow(length(local - (vec2f(hash3(vec3f(cell, 1.0)), hash3(vec3f(cell, 2.0))) - 0.5) * 0.5) - age * 0.45, 2.0) * 900.0) * (1.0 - age);
      let fres = pow(1.0 - clamp(dot(n, normalize(e - world)), 0.0, 1.0), 4.0);
      let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
      c = c + vec3f(ring * 0.08 + fres * 0.12) * w * (0.4 + lum);
    }
  }

  // The fog along the view ray, front to back.
  if (p.fog.x > 0.0) {
    let steps = 14u;
    let dist = min(length(world - e), maxDist);
    let dir = normalize(world - e);
    let stepLen = dist / f32(steps);
    let jitter = ign(in.pos.xy);
    var trans = 1.0;
    var scatter = vec3f(0.0);
    let cosSun = dot(dir, p.sun.xyz);
    let sunPhase = hg(cosSun, p.fogColor.w) * p.sun.w;
    let count = min(p.misc.y, 8u);
    for (var s = 0u; s < steps; s++) {
      let q = e + dir * ((f32(s) + jitter) * stepLen);
      let d = fog_density(q, t);
      if (d <= 0.0) {
        continue;
      }
      var light = p.fogColor.rgb + p.sunColor.rgb * sunPhase * 6.0;
      for (var l = 0u; l < count; l++) {
        let lp = p.lights[l * 2u];
        let lc = p.lights[l * 2u + 1u];
        let to = lp.xyz - q;
        let r = max(lp.w, 0.1);
        let dd = dot(to, to);
        let fall = clamp(1.0 - dd / (r * r), 0.0, 1.0);
        let ph = hg(dot(dir, to * inverseSqrt(max(dd, 1e-4))), p.fogColor.w * 0.5) * 12.566371;
        light += lc.rgb * lc.w * fall * fall * ph / (1.0 + dd * 0.15);
      }
      let ext = d * stepLen;
      let st = exp(-ext);
      scatter += trans * light * (1.0 - st);
      trans *= st;
      if (trans < 0.01) {
        break;
      }
    }
    c = c * trans + scatter;
  }
  return vec4f(srgb_enc(c), f.a);
}

// The sky (Clouds mode, on the sky's pixels): an analytic sky for the time of day (a gradient from
// the horizon to the zenith, a sunset's glow toward the sun, the sun's and the moon's discs and
// halos, stars at night) over the game's painted one, then clouds marched through a slab hundreds
// of metres thick: big shapes from low-frequency noise cut by the coverage, rounded tops, edges
// eroded by finer noise, each step lit by how much cloud lies between it and the sun (or the
// moon) with a powder term and a two-lobe phase, plus the sky's light from above, fading into
// the haze with distance; lightning lights them from inside.
//   lights[0]: direction to the sun (world), its intensity; lights[1]: its colour, its elevation
//   lights[2]: direction to the moon, its intensity;        lights[3]: its colour, w: how far a painted dome left in place is (0: none)
//   lights[4]: how much of the painted sky the analytic one replaces, stars, clouds on, flash
fn sky_color(dir: vec3f) -> vec3f {
  let sunDir = p.lights[0].xyz;
  let sunI = p.lights[0].w;
  let elev = p.lights[1].w;
  let moonDir = p.lights[2].xyz;
  let moonI = p.lights[2].w;
  let up = max(dir.z, 0.0);
  let day = smoothstep(-0.15, 0.25, elev);
  let golden = (1.0 - smoothstep(0.0, 0.35, elev)) * smoothstep(-0.25, 0.02, elev);
  let flatDir = normalize(vec3f(dir.xy, 0.0) + vec3f(1e-5, 0.0, 0.0));
  let flatSun = normalize(vec3f(sunDir.xy, 0.0) + vec3f(1e-5, 0.0, 0.0));
  let toward = pow(dot(flatDir, flatSun) * 0.5 + 0.5, 3.0);
  let zenith = mix(vec3f(0.004, 0.006, 0.016), vec3f(0.11, 0.26, 0.6), day);
  var horizon = mix(vec3f(0.012, 0.016, 0.035), vec3f(0.55, 0.68, 0.86), day);
  horizon = mix(horizon, vec3f(1.0, 0.42, 0.16) * (0.35 + 0.9 * toward), golden);
  var c = mix(horizon, zenith, pow(up, 0.45));
  let cs = dot(dir, sunDir);
  let sunTint = mix(vec3f(1.0, 0.55, 0.25), vec3f(1.0, 0.93, 0.82), smoothstep(0.0, 0.3, elev));
  c += sunTint * (pow(max(cs, 0.0), 8.0) * 0.3 + pow(max(cs, 0.0), 300.0) * 1.2) * sunI;
  c += sunTint * smoothstep(0.99955, 0.9997, cs) * 40.0 * sunI;
  let cm = dot(dir, moonDir);
  c += vec3f(0.8, 0.85, 0.95) * smoothstep(0.99965, 0.99975, cm) * 2.5 * step(0.0, moonI) * smoothstep(-0.02, 0.05, moonDir.z);
  c += vec3f(0.25, 0.3, 0.45) * pow(max(cm, 0.0), 80.0) * 0.4 * moonI;
  let stars = p.lights[4].y;
  if (stars > 0.0 && dir.z > 0.0) {
    let g = dir * 260.0;
    let cell = floor(g);
    let h = hash3(cell);
    if (h > 0.9965) {
      let centre = cell + vec3f(hash3(cell + 7.1), hash3(cell + 13.7), hash3(cell + 3.3));
      let dd = length(g - centre);
      let tw = 0.6 + 0.4 * sin(p.weather.y * (2.0 + h * 5.0) + h * 100.0);
      c += vec3f(0.9, 0.93, 1.0) * smoothstep(0.35, 0.0, dd) * (h - 0.9965) * 400.0 * tw * stars * smoothstep(0.0, 0.2, dir.z);
    }
  }
  return c;
}

fn cloud_density(q: vec3f, e: vec3f, cover: f32, detail: bool) -> f32 {
  let hf = (q.z - e.z - p.clouds.y) / max(p.clouds.z, 1.0);
  if (hf < 0.0 || hf > 1.0) {
    return 0.0;
  }
  let w = q + vec3f(1.0, 0.35, 0.0) * p.clouds.w * p.weather.y;
  let base = fbm(w * vec3f(0.0008, 0.0008, 0.0016), 3u);
  // Each cloud's own height: the tops rise where the shape is strong.
  let topAt = 0.35 + 0.6 * clamp((base - 0.35) * 2.5, 0.0, 1.0);
  let profile = smoothstep(0.0, 0.12, hf) * (1.0 - smoothstep(topAt * 0.7, topAt, hf));
  var d = clamp((base - (1.0 - cover) * 0.75) / max(cover * 0.75, 0.05), 0.0, 1.0) * profile;
  if (d > 0.0 && detail) {
    let er = fbm(w * 0.006 + vec3f(0.0, 0.0, p.weather.y * 0.02), 2u);
    d = max(d - er * 0.45 * (1.0 - d), 0.0);
  }
  return d;
}

@fragment
fn fs_clouds(in: VertexOutput) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(src));
  let px = min(vec2i(floor(in.pos.xy)), size - vec2i(1));
  let f = textureLoad(src, px, 0);
  let dsize = vec2i(textureDimensions(depthTex));
  let dat = min(vec2i(in.uv * vec2f(dsize)), dsize - vec2i(1));
  let z = raw_z(dat);
  // A painted dome left in place: what lies at or past it is its sky.
  let dome = p.lights[3].w;
  let pastDome = dome > 0.0 && !is_viewmodel(z) && linear_of(z) * length(view_ray(in.uv)) >= dome;
  if (!is_sky(z) && !pastDome) {
    return f;
  }
  let dir = normalize(to_world_dir(view_ray(in.uv)));
  // The analytic sky over the painted one (which keeps what is at and below the horizon: its hills).
  let painted = srgb_dec(f.rgb);
  // 2: the painted sky's dome was taken out (nothing behind): all of it.
  let replace = select(p.lights[4].x * smoothstep(0.0, 0.12, dir.z), 1.0, p.lights[4].x > 1.5);
  var skyLin = mix(painted, sky_color(dir), replace);
  if (p.lights[4].z < 0.5 || dir.z < 0.01) {
    return vec4f(srgb_enc(skyLin), f.a);
  }
  let e = eye();
  let base = p.clouds.y;
  let top = base + p.clouds.z;
  let t0 = base / dir.z;
  let t1 = min(top / dir.z, t0 + 5000.0);
  if (t0 > 12000.0) {
    return vec4f(srgb_enc(skyLin), f.a);
  }
  let steps = 40u;
  let stepLen = (t1 - t0) / f32(steps);
  let jitter = ign(in.pos.xy);
  let cover = clamp(p.clouds.x, 0.0, 1.0);
  // The key light: the sun by day, the moon by night.
  let useSun = p.lights[0].w >= p.lights[2].w;
  let keyDir = select(p.lights[2].xyz, p.lights[0].xyz, useSun);
  let keyI = max(select(p.lights[2].w, p.lights[0].w, useSun), 0.0);
  let keyCol = select(p.lights[3].rgb, p.lights[1].rgb, useSun);
  let cosT = dot(dir, keyDir);
  let phase = mix(hg(cosT, 0.65), hg(cosT, -0.15), 0.35) * 12.566371;
  let skyTop = sky_color(normalize(vec3f(0.0, 0.0, 1.0)));
  let skyLow = sky_color(normalize(vec3f(dir.xy, 0.05)));
  let flash = p.lights[4].w;
  var trans = 1.0;
  var light = vec3f(0.0);
  var firstDist = -1.0;
  for (var s = 0u; s < steps; s++) {
    let dist = t0 + (f32(s) + jitter) * stepLen;
    let q = e + dir * dist;
    let d = cloud_density(q, e, cover, true) * p.cloudColor.w;
    if (d <= 0.001) {
      continue;
    }
    if (firstDist < 0.0) {
      firstDist = dist;
    }
    // Toward the light: the cloud it passes through, in a few growing steps.
    var od = 0.0;
    var ls = 40.0;
    var lq = q;
    for (var k = 0u; k < 4u; k++) {
      lq = lq + keyDir * ls;
      od += cloud_density(lq, e, cover, false) * p.cloudColor.w * ls;
      ls *= 1.8;
    }
    let beer = exp(-od * 0.012) * 0.8 + exp(-od * 0.003) * 0.2;
    let powder = 1.0 - exp(-d * stepLen * 0.03);
    let hf = clamp((q.z - e.z - base) / max(p.clouds.z, 1.0), 0.0, 1.0);
    let ambient = mix(skyLow * 0.6, skyTop * 1.1, hf);
    var lit = keyCol * keyI * beer * mix(1.0, powder * 2.0, 0.5) * phase + ambient * 0.9;
    lit = lit * p.cloudColor.rgb;
    lit += vec3f(0.75, 0.8, 1.0) * flash * (0.25 + 0.5 * d);
    let ext = d * stepLen * 0.012;
    let st = exp(-ext);
    light += trans * lit * (1.0 - st);
    trans *= st;
    if (trans < 0.02) {
      break;
    }
  }
  if (firstDist < 0.0) {
    return vec4f(srgb_enc(skyLin), f.a);
  }
  // Into the haze with distance, and thinner toward the horizon.
  let haze = 1.0 - exp(-firstDist / 7000.0);
  let a = (1.0 - trans) * smoothstep(0.01, 0.1, dir.z);
  var cl = light / max(1.0 - trans, 1e-3);
  cl = mix(cl, skyLow, haze);
  let c = mix(skyLin, cl, a);
  return vec4f(srgb_enc(c), f.a);
}

fn project_prev(world: vec3f) -> vec2f {
  // World to the last frame's view: the transpose of its rotation, about its eye.
  let pe = vec3f(p.prevView[0].w, p.prevView[1].w, p.prevView[2].w);
  let d = world - pe;
  let r = mat3x3f(p.prevView[0].xyz, p.prevView[1].xyz, p.prevView[2].xyz);
  let v = r * d;
  let zl = max(-v.z, 1e-4);
  let ray = v.xy / zl;
  let s = (ray - vec2f(p.frustum.x, p.frustum.z)) / vec2f(p.frustum.y - p.frustum.x, p.frustum.w - p.frustum.z);
  return vec2f(s.x, 1.0 - s.y);
}

// Time slowed (misc.w: 0..1000): colour drained toward a cold blue, the edges darkened.
fn slowed(c: vec3f, uv: vec2f) -> vec3f {
  let s = f32(p.misc.w) / 1000.0;
  if (s <= 0.0) {
    return c;
  }
  let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  let cold = vec3f(lum) * vec3f(0.8, 0.95, 1.2);
  let d = length(uv - 0.5) * 1.4;
  let vig = 1.0 - s * 0.55 * smoothstep(0.35, 1.0, d);
  return mix(c, cold, s * 0.65) * vig;
}

@fragment
fn fs_finish(in: VertexOutput) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(src));
  let fsize = vec2f(size);
  let px = min(vec2i(floor(in.pos.xy)), size - vec2i(1));
  let f = textureLoad(src, px, 0);
  let dsize = vec2i(textureDimensions(depthTex));
  let dat = min(vec2i(in.uv * vec2f(dsize)), dsize - vec2i(1));
  let z = raw_z(dat);
  // The time of day's light over the world outdoors (lights[0]: rgb, on), sky excepted.
  let grade = select(vec3f(1.0), p.lights[0].rgb, p.lights[0].w > 0.5 && !is_sky(z));
  if (is_viewmodel(z)) {
    return vec4f(srgb_enc(slowed(srgb_dec(f.rgb) * grade, in.uv)), f.a);
  }
  var zl = p.depth.y;
  if (!is_sky(z)) {
    zl = linear_of(z);
  }
  var c = srgb_dec(f.rgb);

  // Depth of field: a disc as wide as the circle of confusion, each tap weighed by its own.
  if (p.finish.y > 0.0) {
    // A negative focus: what is at the screen's centre (the nearest of a few taps there).
    var focus = p.finish.z;
    if (focus < 0.0) {
      focus = p.depth.y;
      for (var j = 0; j < 5; j++) {
        let o = vec2f(f32(j % 3) - 1.0, f32(j / 3) - 0.5) * 0.04;
        let fz = raw_z(min(vec2i((vec2f(0.5) + o) * vec2f(dsize)), dsize - vec2i(1)));
        if (!is_sky(fz) && !is_viewmodel(fz)) {
          focus = min(focus, linear_of(fz));
        }
      }
    }
    let maxR = 14.0 * p.finish.y * fsize.y / 1080.0;
    let coc = clamp((abs(zl - focus) - p.finish.w * 0.5) / max(p.finish.w, 0.1), 0.0, 1.0) * maxR;
    if (coc > 0.5) {
      var sum = c;
      var wsum = 1.0;
      let jitter = ign(in.pos.xy) * 6.2831853;
      for (var i = 0u; i < 24u; i++) {
        let fi = f32(i) + 0.5;
        let r = sqrt(fi / 24.0) * coc;
        let a = fi * 2.3999632 + jitter;
        let o = vec2f(cos(a), sin(a)) * r;
        let q = clamp(vec2i(in.pos.xy + o), vec2i(0), size - vec2i(1));
        let qz = raw_z(min(vec2i((vec2f(q) + 0.5) / fsize * vec2f(dsize)), dsize - vec2i(1)));
        if (is_viewmodel(qz)) {
          continue;
        }
        var ql = p.depth.y;
        if (!is_sky(qz)) {
          ql = linear_of(qz);
        }
        let qcoc = clamp((abs(ql - focus) - p.finish.w * 0.5) / max(p.finish.w, 0.1), 0.0, 1.0) * maxR;
        // A sharp tap in front doesn't spread over what is blurred behind it.
        let w = select(clamp(qcoc / max(r, 1e-3), 0.0, 1.0), 1.0, ql >= zl - 0.5);
        sum += srgb_dec(textureLoad(src, q, 0).rgb) * w;
        wsum += w;
      }
      c = sum / wsum;
    }
  }

  // Motion blur along where this pixel was in the last frame.
  if (p.finish.x > 0.0) {
    let world = to_world(view_ray(in.uv) * zl);
    let prev = project_prev(world);
    var vel = (in.uv - prev) * p.finish.x * 0.5;
    let len = length(vel * fsize);
    // A cut (or a teleport) is not motion.
    if (len > 1.0 && length(in.uv - prev) < 0.3) {
      let maxPx = 40.0 * fsize.y / 1080.0;
      if (len > maxPx) {
        vel = vel * (maxPx / len);
      }
      let taps = 10u;
      var sum = c;
      var wsum = 1.0;
      let jitter = ign(in.pos.xy) - 0.5;
      for (var i = 1u; i <= taps; i++) {
        let k = (f32(i) + jitter) / f32(taps) - 0.5;
        let q = clamp(vec2i((in.uv + vel * k) * fsize), vec2i(0), size - vec2i(1));
        let qz = raw_z(min(vec2i((vec2f(q) + 0.5) / fsize * vec2f(dsize)), dsize - vec2i(1)));
        if (is_viewmodel(qz)) {
          continue;
        }
        sum += srgb_dec(textureLoad(src, q, 0).rgb);
        wsum += 1.0;
      }
      c = sum / wsum;
    }
  }
  return vec4f(srgb_enc(slowed(c * grade, in.uv)), f.a);
}
)";

struct State {
  EncoderTaskId task = InvalidEncoderTask;
  std::array<wgpu::Buffer, ModeCount> uniforms;
  wgpu::BindGroupLayout layout;
  std::array<wgpu::RenderPipeline, ModeCount> pipelines;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  uint32_t samples = 0;
  wgpu::Texture frame;
  wgpu::TextureView frameView;
  wgpu::TextureFormat frameFormat = wgpu::TextureFormat::Undefined;
  uint32_t width = 0;
  uint32_t height = 0;
  bool logged = false;
};
State g_state;

// The params of each recorded pass: the payload carries a slot in this ring (frames are encoded
// well within its length of being recorded).
std::array<Params, 24> g_recorded;
uint32_t g_nextSlot = 0;

void ensure_pipelines(wgpu::TextureFormat format, uint32_t samples) {
  for (auto& buffer : g_state.uniforms) {
    if (!buffer) {
      const wgpu::BufferDescriptor bufferDescriptor{
          .label = "Post Uniforms",
          .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
          .size = sizeof(Params),
      };
      buffer = g_device.CreateBuffer(&bufferDescriptor);
    }
  }
  if (g_state.pipelines[0] && g_state.format == format && g_state.samples == samples) {
    return;
  }
  const auto stage = wgpu::ShaderStage::Fragment;
  const std::array entries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = stage,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(Params)},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = stage,
          .texture = {.sampleType = wgpu::TextureSampleType::Depth,
                      .viewDimension = wgpu::TextureViewDimension::e2D,
                      .multisampled = samples > 1},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = stage,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "Post Layout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_state.layout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "Post",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_state.layout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  std::string code = Source;
  const std::string token = "DEPTH_TYPE";
  code.replace(code.find(token), token.size(), samples > 1 ? "texture_depth_multisampled_2d" : "texture_depth_2d");
  wgpu::ShaderSourceWGSL source{};
  source.code = code.c_str();
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "Post"};
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);
  const char* entryPoints[ModeCount] = {"fs_atmosphere", "fs_clouds", "fs_finish"};
  const wgpu::ColorTargetState target{.format = format, .writeMask = wgpu::ColorWriteMask::All};
  for (uint32_t m = 0; m < ModeCount; ++m) {
    const wgpu::FragmentState fragment{
        .module = module,
        .entryPoint = entryPoints[m],
        .targetCount = 1,
        .targets = &target,
    };
    const wgpu::RenderPipelineDescriptor descriptor{
        .label = entryPoints[m],
        .layout = pipelineLayout,
        .vertex = {.module = module, .entryPoint = "vs_main"},
        .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
        .multisample = {.count = samples, .mask = UINT32_MAX},
        .fragment = &fragment,
    };
    g_state.pipelines[m] = g_device.CreateRenderPipeline(&descriptor);
  }
  g_state.format = format;
  g_state.samples = samples;
}

void ensure_frame(uint32_t width, uint32_t height, wgpu::TextureFormat format) {
  if (g_state.frame && g_state.width == width && g_state.height == height && g_state.frameFormat == format) {
    return;
  }
  g_state.width = width;
  g_state.height = height;
  g_state.frameFormat = format;
  const wgpu::TextureDescriptor descriptor{
      .label = "Post Frame Copy",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .size = {width, height, 1},
      .format = format,
  };
  g_state.frame = g_device.CreateTexture(&descriptor);
  g_state.frameView = g_state.frame.CreateView();
}

void encode(const EncoderTaskContext& ctx, const wgpu::CommandEncoder& cmd, const void* payload, size_t payloadSize,
            void*) {
  uint32_t slot = 0;
  if (payloadSize != sizeof(slot)) {
    return;
  }
  std::memcpy(&slot, payload, sizeof(slot));
  const Params params = g_recorded[slot % g_recorded.size()];
  const uint32_t mode = params.misc[0];
  if (mode >= ModeCount) {
    return;
  }
  const auto& source = webgpu::present_source();
  const auto& target = webgpu::g_frameBuffer;
  const auto& depth = webgpu::g_depthBuffer;
  const uint32_t samples = webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_graphicsConfig.msaaSamples : 1;
  const auto format = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  const uint32_t width = source.size.width;
  const uint32_t height = source.size.height;
  if (width == 0 || height == 0 || !source.texture || !target.view || !depth.view) {
    return;
  }
  ensure_pipelines(format, samples);
  ensure_frame(width, height, format);
  if (!g_state.logged) {
    g_state.logged = true;
    Log.info("post passes ready ({}x{}, {} samples)", width, height, samples);
  }
  ctx.queue.WriteBuffer(g_state.uniforms[mode], 0, &params, sizeof(params));

  const wgpu::TexelCopyTextureInfo copySource{.texture = source.texture};
  const wgpu::TexelCopyTextureInfo copyTarget{.texture = g_state.frame};
  const wgpu::Extent3D copySize{width, height, 1};
  cmd.CopyTextureToTexture(&copySource, &copyTarget, &copySize);

  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = g_state.uniforms[mode], .size = sizeof(Params)},
      wgpu::BindGroupEntry{.binding = 1, .textureView = depth.view},
      wgpu::BindGroupEntry{.binding = 2, .textureView = g_state.frameView},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "Post",
      .layout = g_state.layout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto group = g_device.CreateBindGroup(&groupDescriptor);
  static const char* const labels[ModeCount] = {"Post atmosphere", "Post clouds", "Post finish"};
  const wgpu::RenderPassColorAttachment attachment{
      .view = target.view,
      .resolveTarget = samples > 1 ? webgpu::g_frameBufferResolved.view : wgpu::TextureView{},
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
  };
  const wgpu::RenderPassDescriptor passDescriptor{
      .label = labels[mode],
      .colorAttachmentCount = 1,
      .colorAttachments = &attachment,
      .timestampWrites = webgpu::gpu_prof::pass_writes(labels[mode]),
  };
  const auto pass = cmd.BeginRenderPass(&passDescriptor);
  pass.SetPipeline(g_state.pipelines[mode]);
  pass.SetBindGroup(0, group);
  pass.Draw(3);
  pass.End();
}
} // namespace

bool ensure_task() {
  if (g_state.task == InvalidEncoderTask) {
    g_state.task = register_encoder_task_type(EncoderTaskDescriptor{.label = "Post", .callback = encode});
    if (g_state.task == InvalidEncoderTask) {
      Log.warn("could not register the post task");
      return false;
    }
  }
  return true;
}

bool record(const Params& params) {
  if (g_state.task == InvalidEncoderTask || params.misc[0] >= ModeCount) {
    return false;
  }
  const auto& size = webgpu::present_source().size;
  if (size.width == 0 || size.height == 0) {
    return false;
  }
  const uint32_t slot = g_nextSlot++ % g_recorded.size();
  g_recorded[slot] = params;
  return record_encoder_task(g_state.task, &slot, sizeof(slot));
}

void shutdown() {
  const auto task = g_state.task;
  g_state = {};
  if (task != InvalidEncoderTask) {
    unregister_encoder_task_type(task);
  }
}
} // namespace aurora::gfx::post
