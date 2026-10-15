#include <aurora/fxp.hpp>

#include "../gx/fifo.hpp"
#include "../gx/gx.hpp"
#include "../logging.hpp"
#include "../webgpu/gpu.hpp"
#include "recording.hpp"
#include "scenecap.hpp"

#include <aurora/gfx.hpp>

#include <array>
#include <cstring>
#include <map>
#include <vector>

// The GPU particle layer (see aurora/fxp.hpp).
namespace aurora::gfx::fxp {
namespace {
Module Log("aurora::gfx::fxp");

constexpr uint32_t GridVec4 = Field::GridSize * Field::GridSize / 4;

struct Uniform {
  float proj[16];
  float mv[12];
  float info[4];   // time, anchor pulse, scene depth valid, 0
  float anchor[4]; // the attached particles' centre
  float fieldA[4]; // radius, height, speed, size
  float fieldB[4]; // length, wind xyz
  float fieldC[4]; // grid origin x, y, cell, use grid
  float fieldD[4]; // kind, 0, 0, 0
  float color[4];  // the field's colour, rgba
  float sun[4];    // the modern materials' sun (gx::GXState::modernSun): view-space travel, intensity
  float grid[GridVec4][4];
};
static_assert(sizeof(Uniform) == 16 * 4 + 12 * 4 + 8 * 16 + GridVec4 * 16);

// AdditiveOver: light over everything (the charge at the muzzle, which the gun's nearer depth range
// would otherwise hide).
enum Blend : uint32_t { Additive, Blended, HazeBlend, AdditiveOver, BlendCount };
enum Kind : uint32_t { KindParticles, KindField };

struct Payload {
  uint32_t kind;
  uint32_t blend;
  uint32_t count;
  uint32_t pad;
  Range uniform;
  Range instances;
};
static_assert(sizeof(Payload) <= InlineDrawPayloadSize);

constexpr const char* ShaderSource = R"(
struct U {
  proj: mat4x4f,
  mv: mat3x4f,
  info: vec4f,
  anchor: vec4f,
  fieldA: vec4f,
  fieldB: vec4f,
  fieldC: vec4f,
  fieldD: vec4f,
  color: vec4f,
  sun: vec4f,
  grid: array<vec4f, 256>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var scene: texture_3d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f,
  @location(2) info: vec4f, // shape, life fraction, seed, view depth
};

const CORNERS = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                                vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));

fn hash(n: u32) -> f32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  return f32(x) / 4294967295.0;
}

fn unpack(c: u32) -> vec4f {
  return vec4f(f32((c >> 16u) & 255u), f32((c >> 8u) & 255u), f32(c & 255u), f32((c >> 24u) & 255u)) / 255.0;
}

fn to_view(w: vec3f) -> vec3f {
  return vec4f(w, 1.0) * u.mv;
}

fn eye() -> vec3f {
  let r0 = u.mv[0];
  let r1 = u.mv[1];
  let r2 = u.mv[2];
  return -vec3f(r0.x * r0.w + r1.x * r1.w + r2.x * r2.w, r0.y * r0.w + r1.y * r1.w + r2.y * r2.w,
                r0.z * r0.w + r1.z * r1.w + r2.z * r2.w);
}

fn rotate_axis(v: vec3f, axis: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

// A quad at world point `pos`: a billboard, a streak along the world velocity `vel`, or flat on
// the ground (print).
fn quad(corner: vec2f, pos: vec3f, vel: vec3f, size: f32, stretch: f32, mode: u32, seed: f32) -> vec4f {
  var vp: vec3f;
  if (mode == 2u) {
    let yaw = seed * 6.2831853;
    let ax = vec3f(cos(yaw), sin(yaw), 0.0);
    let ay = vec3f(-ax.y, ax.x, 0.0);
    vp = to_view(pos + ax * (corner.x * size * 0.35) + ay * (corner.y * size * 0.5) + vec3f(0.0, 0.0, 0.03));
  } else {
    vp = to_view(pos);
    if (mode == 1u) {
      let vv = to_view(pos + vel * 0.01) - vp;
      var dir = vec2f(0.0, 1.0);
      if (dot(vv.xy, vv.xy) > 1e-10) {
        dir = normalize(vv.xy);
      }
      let side = vec2f(-dir.y, dir.x);
      let len = size * 0.5 + length(vel) * stretch * 0.5;
      vp = vp + vec3f(dir * (corner.y * len) + side * (corner.x * size * 0.5), 0.0);
    } else {
      vp = vp + vec3f(corner * (size * 0.5), 0.0);
    }
  }
  return vec4f(vp, 1.0) * u.proj;
}

fn shape_mode(shape: u32) -> u32 {
  if (shape == 6u) { return 2u; }
  if (shape == 1u || shape == 3u || shape == 8u) { return 1u; }
  return 0u;
}

@vertex
fn vs_particle(@builtin(vertex_index) vi: u32, @location(0) p0t0: vec4f, @location(1) v0life: vec4f,
               @location(2) phys: vec4f, @location(3) cf: vec2<u32>, @location(4) extra: vec2f) -> VOut {
  var out: VOut;
  let age = u.info.x - p0t0.w;
  let life = max(v0life.w, 1e-3);
  if (age < 0.0 || age > life) {
    out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
    return out;
  }
  let f = age / life;
  let shape = cf.y & 255u;
  let seed = extra.y;
  var pos: vec3f;
  var vel: vec3f;
  if ((cf.y & 256u) != 0u) {
    // Spiralling in on the anchor.
    let axis = normalize(v0life.xyz + vec3f(1e-5, 0.0, 0.0));
    let off = rotate_axis(p0t0.xyz * pow(1.0 - f, 1.5), axis, seed * 6.2831853 + age * (5.0 + seed * 7.0));
    pos = u.anchor.xyz + off;
    vel = -off * 3.0;
  } else {
    let k = max(phys.y, 1e-3);
    let g = vec3f(0.0, 0.0, -phys.z);
    let e = exp(-k * age);
    pos = p0t0.xyz + g / k * age + (v0life.xyz - g / k) * (1.0 - e) / k;
    vel = g / k + (v0life.xyz - g / k) * e;
    if (pos.z < phys.w) {
      pos.z = phys.w + (phys.w - pos.z) * 0.22;
      vel.z = -vel.z * 0.22;
    }
  }
  var size = phys.x;
  if (shape == 2u) {
    size = size * (0.6 + 1.7 * f);
  } else if (shape == 5u) {
    size = size * (0.15 + 1.85 * f);
  }
  if ((cf.y & 512u) != 0u) {
    size = size * (1.0 - f);
  }
  let corner = CORNERS[vi];
  out.pos = quad(corner, pos, vel, size, extra.x, shape_mode(shape), seed);
  out.uv = corner;
  var c = unpack(cf.x);
  if ((cf.y & 256u) != 0u) {
    c = vec4f(c.rgb * (1.0 + 1.5 * u.anchor.w), c.a);
  }
  out.color = c;
  out.info = vec4f(f32(shape), f, seed, -to_view(pos).z);
  return out;
}

@vertex
fn vs_field(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
  let kind = u32(u.fieldD.x);
  let t = u.info.x;
  let h = vec3f(hash(ii * 3u + 1u), hash(ii * 3u + 2u), hash(ii * 3u + 3u));
  let r = u.fieldA.x;
  let box = vec3f(2.0 * r, 2.0 * r, u.fieldA.y);
  let cam = eye();
  let lo = cam - box * 0.5;
  var pos: vec3f;
  var vel: vec3f;
  var shape = 3u;
  var size = u.fieldA.w;
  var fade = 1.0;
  if (kind == 4u) {
    // A splash on the grid's top surface, over and over at its own pace.
    let cell = u.fieldC.z;
    let xy = vec2f(u.fieldC.x, u.fieldC.y) + h.xy * cell * 32.0;
    let gi = vec2i(clamp(floor((xy - u.fieldC.xy) / cell), vec2f(0.0), vec2f(31.0)));
    let idx = u32(gi.y * 32 + gi.x);
    let top = u.grid[idx / 4u][idx % 4u];
    if (top < -1e8 || distance(xy, cam.xy) > r) {
      return out;
    }
    let cycle = 0.25 + 0.35 * h.z;
    let local = fract((t + h.z * 17.0) / cycle);
    pos = vec3f(xy, top + 0.02);
    vel = vec3f(0.0);
    shape = 5u;
    size = u.fieldA.w * (0.3 + 1.2 * local);
    fade = 1.0 - local;
  } else {
    var drift = vec3f(u.fieldB.yzw * t);
    if (kind == 0u) {
      drift.z = drift.z - u.fieldA.z * t * (0.85 + 0.3 * h.z);
      vel = vec3f(u.fieldB.yz, -u.fieldA.z);
      shape = 3u;
    } else if (kind == 1u) {
      drift.z = drift.z - u.fieldA.z * t * (0.7 + 0.6 * h.x);
      drift.x = drift.x + sin(t * 1.3 + h.y * 30.0) * 0.4;
      drift.y = drift.y + cos(t * 1.1 + h.x * 30.0) * 0.4;
      vel = vec3f(0.0);
      shape = 4u;
      size = size * (0.6 + 0.8 * h.y);
    } else if (kind == 2u) {
      drift = drift + vec3f(sin(t * 0.21 + h.x * 40.0), cos(t * 0.17 + h.y * 40.0), sin(t * 0.13 + h.z * 40.0)) * 0.6;
      vel = vec3f(0.0);
      shape = 0u;
      size = size * (0.5 + h.z);
    } else {
      drift.z = drift.z + u.fieldA.z * t * (0.6 + 0.8 * h.z);
      drift.x = drift.x + sin(t * 2.1 + h.y * 50.0) * 0.5;
      vel = vec3f(0.0);
      shape = 10u;
      size = size * (0.5 + h.x);
    }
    pos = lo + ((h * box + drift - lo) % box + box) % box;
    if (u.fieldC.w > 0.5) {
      let gi = vec2i(floor((pos.xy - u.fieldC.xy) / u.fieldC.z));
      if (all(gi >= vec2i(0)) && all(gi < vec2i(32))) {
        let idx = u32(gi.y * 32 + gi.x);
        if (pos.z < u.grid[idx / 4u][idx % 4u]) {
          return out;
        }
      }
    }
    let d = distance(pos, cam);
    fade = smoothstep(0.4, 2.0, d) * (1.0 - smoothstep(r * 0.7, r, d));
  }
  let corner = CORNERS[vi];
  out.pos = quad(corner, pos, vel, size, u.fieldB.x / max(u.fieldA.z, 1e-3), shape_mode(shape), h.x);
  out.uv = corner;
  out.color = vec4f(u.color.rgb, u.color.a * fade);
  out.info = vec4f(f32(shape), 0.5, h.y, -to_view(pos).z);
  return out;
}

fn wobble(p: vec2f) -> f32 {
  return sin(p.x * 3.1 + sin(p.y * 2.3)) * sin(p.y * 2.7 + sin(p.x * 1.9)) * 0.5 + 0.5;
}

struct Shaped {
  rgb: vec3f,
  a: f32,
};

fn shape_of(in: VOut) -> Shaped {
  let shape = u32(in.info.x + 0.5);
  let f = in.info.y;
  let seed = in.info.z;
  let uv = in.uv;
  let r = length(uv);
  var a = 0.0;
  var rgb = in.color.rgb;
  let fadeIn = smoothstep(0.0, 0.08, f);
  let fadeOut = 1.0 - smoothstep(0.65, 1.0, f);
  switch (shape) {
    case 0u: {
      a = exp(-r * r * 3.5) * fadeIn * fadeOut;
    }
    case 10u: {
      a = exp(-r * r * 4.0) * fadeIn * fadeOut * (0.55 + 0.45 * sin(u.info.x * 17.0 + seed * 61.0));
      rgb = rgb * (1.0 + exp(-r * r * 14.0));
    }
    case 1u: {
      a = exp(-uv.x * uv.x * 5.0) * (1.0 - uv.y * uv.y) * fadeOut;
      rgb = rgb + vec3f(exp(-uv.x * uv.x * 30.0) * 0.6);
    }
    case 2u: {
      let n = wobble(uv * 2.2 + vec2f(seed * 31.0, f * 1.7)) * 0.6 + wobble(uv * 4.7 - vec2f(f, seed * 17.0)) * 0.4;
      a = smoothstep(1.0, 0.15, r) * (0.45 + 0.55 * n) * fadeIn * (1.0 - f);
    }
    case 3u: {
      a = exp(-uv.x * uv.x * 6.0) * smoothstep(1.0, 0.2, abs(uv.y)) * fadeOut;
    }
    case 4u: {
      a = smoothstep(1.0, 0.25, r) * fadeOut;
    }
    case 5u: {
      a = exp(-(r - 0.8) * (r - 0.8) * 90.0) * (1.0 - f) * fadeIn;
    }
    case 6u: {
      let q = uv * vec2f(1.0, 1.0);
      a = smoothstep(1.0, 0.75, length(q)) * (1.0 - smoothstep(0.6, 1.0, f));
      rgb = rgb * 0.6;
    }
    case 8u: {
      let off = sin(uv.y * 9.0 + u.info.x * 45.0 + seed * 20.0) * 0.35 + sin(uv.y * 23.0 - u.info.x * 70.0) * 0.15;
      let d = uv.x - off;
      a = exp(-d * d * 40.0) * fadeOut;
      rgb = rgb + vec3f(exp(-d * d * 200.0));
    }
    case 9u: {
      a = (exp(-(r - 0.8) * (r - 0.8) * 60.0) * 0.8 + exp(-dot(uv - vec2f(-0.3, 0.35), uv - vec2f(-0.3, 0.35)) * 40.0) * 0.6) *
          fadeOut * step(r, 1.0);
      rgb = vec3f(0.8, 0.9, 1.0);
    }
    default: {
      a = 0.0;
    }
  }
  // Port extension (the modern materials' sun, u.sun): smoke lit as a volume (a sphere's normal
  // across the quad, wrapped, darker away from the light, a silver lining against the sun), and
  // drops and flakes glinting when the light comes from behind them.
  let sunOn = u.sun.w > 0.0;
  let sl = select(normalize(vec3f(0.2, 0.9, 0.35)), -normalize(u.sun.xyz), sunOn);
  let sI = select(0.7, u.sun.w, sunOn);
  if (shape == 2u) {
    let sn = normalize(vec3f(uv.x, uv.y, sqrt(max(1.0 - r * r, 0.05))));
    let wrap = clamp((dot(sn, sl) + 0.6) / 1.6, 0.0, 1.0);
    let lining = pow(clamp(-sl.z, 0.0, 1.0), 6.0) * smoothstep(0.4, 1.0, r) * 1.4;
    rgb = rgb * (0.55 + (0.6 * wrap + lining) * sI);
  } else if (shape == 3u || shape == 4u) {
    let glint = pow(clamp(-sl.z, 0.0, 1.0), 8.0) * sI;
    rgb = rgb + vec3f(1.0, 0.97, 0.9) * glint * 0.8;
  }
  return Shaped(rgb, clamp(a, 0.0, 1.0) * in.color.a);
}

fn soft(in: VOut) -> f32 {
  if (u.info.z < 0.5) {
    return 1.0;
  }
  let dims = vec2i(textureDimensions(scene).xy);
  let px = clamp(vec2i(in.pos.xy * vec2f(dims) / max(u.fieldD.zw, vec2f(1.0))), vec2i(0), dims - 1);
  let sd = textureLoad(scene, vec3i(px, 0), 0).a;
  return clamp((sd - in.info.w) / 0.35, 0.0, 1.0);
}

@fragment
fn fs_add(in: VOut) -> @location(0) vec4f {
  let s = shape_of(in);
  let a = s.a * soft(in);
  if (a < 0.002) { discard; }
  return vec4f(s.rgb * a, 0.0);
}

@fragment
fn fs_blend(in: VOut) -> @location(0) vec4f {
  let s = shape_of(in);
  let a = s.a * soft(in);
  if (a < 0.002) { discard; }
  return vec4f(s.rgb * a, a);
}

@fragment
fn fs_haze(in: VOut) -> @location(0) vec4f {
  if (u.info.z < 0.5) { discard; }
  let f = in.info.y;
  let r = length(in.uv);
  let w = smoothstep(1.0, 0.2, r) * (1.0 - f) * smoothstep(0.0, 0.15, f) * in.color.a;
  if (w < 0.01) { discard; }
  let dims = vec2i(textureDimensions(scene).xy);
  let t = u.info.x;
  let off = vec2f(sin(in.uv.y * 9.0 + t * 7.0 + in.info.z * 20.0), cos(in.uv.x * 8.0 - t * 9.0)) * 6.0 * w;
  let base = in.pos.xy * vec2f(dims) / max(u.fieldD.zw, vec2f(1.0));
  let px = clamp(vec2i(base + off), vec2i(0), dims - 1);
  let c = textureLoad(scene, vec3i(px, 0), 0);
  if (c.a < in.info.w - 0.2) { discard; }
  return vec4f(c.rgb * w, w);
}
)";

struct State {
  DrawTypeId drawType = InvalidDrawType;
  wgpu::BindGroupLayout bindLayout;
  wgpu::PipelineLayout pipelineLayout;
  wgpu::ShaderModule module;
  std::map<std::array<uint64_t, 5>, wgpu::RenderPipeline> pipelines;
  std::map<std::pair<const void*, const void*>, wgpu::BindGroup> groups;
};
State g_state;

void ensure_static(const wgpu::Device& device) {
  if (g_state.module) {
    return;
  }
  const std::array entries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
          .buffer = {.type = wgpu::BufferBindingType::Uniform,
                     .hasDynamicOffset = true,
                     .minBindingSize = sizeof(Uniform)},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e3D},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "FXP Bind Group Layout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_state.bindLayout = device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "FXP Pipeline Layout",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_state.bindLayout,
  };
  g_state.pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  wgpu::ShaderSourceWGSL wgsl{};
  wgsl.code = ShaderSource;
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "FXP Module"};
  g_state.module = device.CreateShaderModule(&moduleDescriptor);
}

wgpu::RenderPipeline pipeline_for(const DrawContext& ctx, uint32_t kind, uint32_t blend) {
  const auto& layout = ctx.layout;
  const std::array<uint64_t, 5> key{layout.key, layout.sampleCount, kind, blend,
                                    static_cast<uint64_t>(layout.depthStencilFormat)};
  const auto it = g_state.pipelines.find(key);
  if (it != g_state.pipelines.end()) {
    return it->second;
  }
  const std::array<wgpu::VertexAttribute, 5> attributes{{
      {.format = wgpu::VertexFormat::Float32x4, .offset = 0, .shaderLocation = 0},
      {.format = wgpu::VertexFormat::Float32x4, .offset = 16, .shaderLocation = 1},
      {.format = wgpu::VertexFormat::Float32x4, .offset = 32, .shaderLocation = 2},
      {.format = wgpu::VertexFormat::Uint32x2, .offset = 48, .shaderLocation = 3},
      {.format = wgpu::VertexFormat::Float32x2, .offset = 56, .shaderLocation = 4},
  }};
  const wgpu::VertexBufferLayout vertexLayout{
      .stepMode = wgpu::VertexStepMode::Instance,
      .arrayStride = sizeof(Particle),
      .attributeCount = attributes.size(),
      .attributes = attributes.data(),
  };
  const wgpu::BlendState add{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::One},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const wgpu::BlendState over{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  std::array<wgpu::ColorTargetState, MaxColorAttachments> targets{};
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    const bool sceneTarget = i == SceneColorAttachmentIndex;
    targets[i] = {
        .format = layout.colorAttachments[i].format,
        .blend = sceneTarget ? (blend == Additive || blend == AdditiveOver ? &add : &over) : nullptr,
        .writeMask = sceneTarget ? wgpu::ColorWriteMask::All : wgpu::ColorWriteMask::None,
    };
  }
  const char* fragmentEntry =
      blend == Additive || blend == AdditiveOver ? "fs_add" : blend == Blended ? "fs_blend" : "fs_haze";
  const wgpu::FragmentState fragment{
      .module = g_state.module,
      .entryPoint = fragmentEntry,
      .targetCount = layout.colorAttachmentCount,
      .targets = targets.data(),
  };
  const wgpu::DepthStencilState depth{
      .format = layout.depthStencilFormat,
      .depthWriteEnabled = false,
      .depthCompare = blend == AdditiveOver ? wgpu::CompareFunction::Always
                      : gx::UseReversedZ        ? wgpu::CompareFunction::GreaterEqual
                                                : wgpu::CompareFunction::LessEqual,
  };
  const bool particles = kind == KindParticles;
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = particles ? "FXP Particles" : "FXP Field",
      .layout = g_state.pipelineLayout,
      .vertex = {.module = g_state.module,
                 .entryPoint = particles ? "vs_particle" : "vs_field",
                 .bufferCount = particles ? 1u : 0u,
                 .buffers = particles ? &vertexLayout : nullptr},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList, .cullMode = wgpu::CullMode::None},
      .depthStencil = layout.depthStencilFormat != wgpu::TextureFormat::Undefined ? &depth : nullptr,
      .multisample = {.count = layout.sampleCount, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  auto pipeline = ctx.device.CreateRenderPipeline(&descriptor);
  g_state.pipelines.emplace(key, pipeline);
  return pipeline;
}

void draw_payload(const DrawContext& ctx, const wgpu::RenderPassEncoder& pass, const void* data, size_t size, void*) {
  if (size != sizeof(Payload)) {
    return;
  }
  Payload p;
  std::memcpy(&p, data, sizeof(p));
  ensure_static(ctx.device);
  const wgpu::TextureView& sceneView = scenecap::view();
  if (!sceneView || p.count == 0) {
    return;
  }
  const std::pair<const void*, const void*> groupKey{ctx.uniformBuffer.Get(), sceneView.Get()};
  auto group = g_state.groups.find(groupKey);
  if (group == g_state.groups.end()) {
    if (g_state.groups.size() > 64) {
      g_state.groups.clear();
    }
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .buffer = ctx.uniformBuffer, .size = sizeof(Uniform)},
        wgpu::BindGroupEntry{.binding = 1, .textureView = sceneView},
    };
    const wgpu::BindGroupDescriptor descriptor{
        .label = "FXP Bind Group",
        .layout = g_state.bindLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    group = g_state.groups.emplace(groupKey, ctx.device.CreateBindGroup(&descriptor)).first;
  }
  pass.SetPipeline(pipeline_for(ctx, p.kind, p.blend));
  const uint32_t dynamicOffset = p.uniform.offset;
  pass.SetBindGroup(0, group->second, 1, &dynamicOffset);
  if (p.kind == KindParticles) {
    pass.SetVertexBuffer(0, ctx.vertexBuffer, p.instances.offset, p.instances.size);
  }
  pass.Draw(6, p.count);
}

bool ensure_type() {
  if (g_state.drawType == InvalidDrawType) {
    g_state.drawType = register_draw_type(DrawTypeDescriptor{.label = "FXP", .draw = draw_payload});
  }
  return g_state.drawType != InvalidDrawType;
}

void fill_common(Uniform& u, float time, bool sceneDepth) {
  const auto& gx = gx::g_gxState;
  Mat4x4<float> proj = gx.proj;
  if (gx::UseReversedZ) {
    proj.m2 = proj.m2 * Vec4<float>{-1.f, -1.f, -1.f, -1.f};
  } else {
    proj.m2 = proj.m2 + proj.m3;
  }
  std::memcpy(u.proj, &proj, sizeof(u.proj));
  std::memcpy(u.mv, &gx.pnMtx[gx.currentPnMtx].pos, sizeof(u.mv));
  u.info[0] = time;
  u.info[2] = sceneDepth ? 1.f : 0.f;
  if (gx.modern > 0.f) {
    std::memcpy(u.sun, &gx.modernSun, sizeof(u.sun));
  }
  // The frame's size: the scene capture may be smaller (a 3D texture's limit).
  const auto& frame = webgpu::g_frameBuffer.size;
  u.fieldD[2] = static_cast<float>(frame.width);
  u.fieldD[3] = static_cast<float>(frame.height);
}

void record(uint32_t kind, uint32_t blend, uint32_t count, const Uniform& u, const Particle* particles) {
  Payload p{};
  p.kind = kind;
  p.blend = blend;
  p.count = count;
  p.uniform = push_uniform(reinterpret_cast<const uint8_t*>(&u), sizeof(u));
  if (overflowed(p.uniform) || p.uniform.size == 0) {
    return;
  }
  if (particles != nullptr) {
    p.instances = push_verts(reinterpret_cast<const uint8_t*>(particles), size_t(count) * sizeof(Particle), 4);
    if (overflowed(p.instances) || p.instances.size == 0) {
      return;
    }
  }
  push_custom_draw(g_state.drawType, &p, sizeof(p));
}
} // namespace

void draw(const Frame& frame, const Particle* particles, uint32_t count) {
  if (particles == nullptr || count == 0 || !ensure_type()) {
    return;
  }
  // The recorder and the GX state belong to the FIFO thread (see geo.cpp).
  gx::fifo::drain();
  static Uniform u;
  std::memset(&u, 0, sizeof(u));
  fill_common(u, frame.time, frame.sceneDepth);
  u.info[1] = frame.anchorPulse;
  u.anchor[0] = frame.anchor[0];
  u.anchor[1] = frame.anchor[1];
  u.anchor[2] = frame.anchor[2];
  u.anchor[3] = frame.anchorPulse;
  // Split by how each blends: light adds, smoke and water lie over, the haze bends.
  static std::vector<Particle> buckets[BlendCount];
  for (auto& b : buckets) {
    b.clear();
  }
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t shape = particles[i].flags & 255u;
    const bool additive = shape == Glow || shape == Spark || shape == Ring || shape == Arc || shape == Ember;
    const bool attached = (particles[i].flags & Attached) != 0;
    const uint32_t bucket = shape == Haze ? HazeBlend : additive ? (attached ? AdditiveOver : Additive) : Blended;
    if (bucket == HazeBlend && !frame.sceneDepth) {
      continue;
    }
    buckets[bucket].push_back(particles[i]);
  }
  // The haze first (it shows the frame captured before any of these), then smoke and drops, then
  // light over them.
  for (const uint32_t b : {HazeBlend, Blended, Additive, AdditiveOver}) {
    if (!buckets[b].empty()) {
      record(KindParticles, b, static_cast<uint32_t>(buckets[b].size()), u, buckets[b].data());
    }
  }
}

void draw_field(const Field& field) {
  if (field.count == 0 || !ensure_type()) {
    return;
  }
  gx::fifo::drain();
  static Uniform u;
  std::memset(&u, 0, sizeof(u));
  fill_common(u, field.time, field.sceneDepth);
  u.fieldA[0] = field.radius;
  u.fieldA[1] = field.height;
  u.fieldA[2] = field.speed;
  u.fieldA[3] = field.size;
  u.fieldB[0] = field.length;
  u.fieldB[1] = field.wind[0];
  u.fieldB[2] = field.wind[1];
  u.fieldB[3] = field.wind[2];
  u.fieldC[0] = field.originX;
  u.fieldC[1] = field.originY;
  u.fieldC[2] = field.cell;
  u.fieldC[3] = field.useGrid ? 1.f : 0.f;
  u.fieldD[0] = static_cast<float>(field.kind);
  u.color[0] = float((field.color >> 16) & 255) / 255.f;
  u.color[1] = float((field.color >> 8) & 255) / 255.f;
  u.color[2] = float(field.color & 255) / 255.f;
  u.color[3] = float((field.color >> 24) & 255) / 255.f;
  std::memcpy(u.grid, field.grid, sizeof(u.grid));
  const uint32_t blend = field.kind == Embers ? Additive : Blended;
  record(KindField, blend, field.count, u, nullptr);
}

void shutdown() {
  const auto type = g_state.drawType;
  g_state = {};
  if (type != InvalidDrawType) {
    unregister_draw_type(type);
  }
}
} // namespace aurora::gfx::fxp
