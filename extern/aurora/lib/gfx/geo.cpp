#include <aurora/geo.hpp>

#include "../gx/fifo.hpp"
#include "../gx/gx.hpp"
#include "../logging.hpp"
#include "../webgpu/gpu.hpp"
#include "recording.hpp"

#include <aurora/gfx.hpp>

#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

// GPU-built geometry (see aurora/geo.hpp): instanced draws whose vertex shader makes each blade
// or rock from its instance record and the vertex index, the way a mesh shader would.
namespace aurora::gfx::geo {
namespace {
Module Log("aurora::gfx::geo");

struct Uniform {
  float proj[16];
  float mv[12];
  float params[4]; // time, wind x, wind y, wind strength
  float fade[4];   // fade start, fade end, pusher count, 0
  float pushers[GrassDraw::MaxPushers][4];
  float fogColor[4];
  float fog[4]; // a, b, c, type
};
static_assert(sizeof(Uniform) == 304);

enum Kind : uint32_t { KindGrass, KindRocks };

struct Payload {
  uint32_t kind;
  uint32_t count;
  uint32_t first;
  uint32_t pad;
  uint64_t key;
  Range uniform;
  Range instances; // rocks: in the frame's vertex buffer
};
static_assert(sizeof(Payload) <= InlineDrawPayloadSize);

constexpr const char* ShaderSource = R"(
struct U {
  proj: mat4x4f,
  mv: mat3x4f,
  params: vec4f,
  fade: vec4f,
  pushers: array<vec4f, 8>,
  fogColor: vec4f,
  fog: vec4f,
};
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
};

fn unpack(c: u32) -> vec3f {
  return vec3f(f32((c >> 16u) & 255u), f32((c >> 8u) & 255u), f32(c & 255u)) / 255.0;
}

fn hash(n: u32) -> f32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  return f32(x) / 4294967295.0;
}

fn to_clip(world: vec3f) -> vec4f {
  let vp = vec4f(world, 1.0) * u.mv;
  return vec4f(vp, 1.0) * u.proj;
}

// A blade: three segments, the last a tip, as five triangles (15 vertices): (level, side).
const BLADE = array<vec2f, 15>(
  vec2f(0.0, -1.0), vec2f(0.0, 1.0), vec2f(1.0, -1.0),
  vec2f(1.0, -1.0), vec2f(0.0, 1.0), vec2f(1.0, 1.0),
  vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(2.0, -1.0),
  vec2f(2.0, -1.0), vec2f(1.0, 1.0), vec2f(2.0, 1.0),
  vec2f(2.0, -1.0), vec2f(2.0, 1.0), vec2f(3.0, 0.0));

@vertex
fn vs_grass(@builtin(vertex_index) vi: u32, @location(0) rootYaw: vec4f, @location(1) size: vec2f,
            @location(2) color: u32, @location(3) seed: u32) -> VOut {
  var out: VOut;
  let root = rootYaw.xyz;
  let dist = length(vec4f(root, 1.0) * u.mv);
  let k = 1.0 - smoothstep(u.fade.x, u.fade.y, dist);
  if (k <= 0.0) {
    out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
    out.color = vec3f(0.0);
    return out;
  }
  let lv = BLADE[vi];
  let t = lv.x / 3.0;
  let h = size.x * (0.35 + 0.65 * k);
  let right = vec3f(cos(rootYaw.w), sin(rootYaw.w), 0.0);
  let fwd = vec3f(-right.y, right.x, 0.0);
  let phase = hash(seed) * 6.2831853;
  // Its own curl, the wind's sway and gusts, and what pushes through it.
  var bend = fwd.xy * (0.25 + 0.35 * hash(seed + 7u)) * h;
  let wdir = u.params.yz;
  let gust = sin(u.params.x * 1.3 + dot(root.xy, wdir) * 0.21) * 0.5 + 0.5;
  bend = bend + wdir * (sin(u.params.x * 2.3 + phase + dot(root.xy, wdir) * 0.6) * 0.18 + gust * 0.32) *
         u.params.w * h;
  var press = 0.0;
  for (var i = 0u; i < min(u32(u.fade.z), 8u); i++) {
    let p = u.pushers[i];
    let d = root.xy - p.xy;
    let dl = length(d);
    if (dl < p.w && abs(root.z - p.z) < 3.0) {
      let a = 1.0 - dl / p.w;
      bend = bend + select(vec2f(0.0), d / dl, dl > 1e-3) * a * 1.1 * h;
      press = max(press, a);
    }
  }
  let bl = length(bend);
  let maxBend = 0.92 * h;
  bend = bend * min(1.0, maxBend / max(bl, 1e-4));
  // The tip keeps the blade's length: the more it bends, the lower it stands.
  let lean = min(length(bend) / h, 0.92);
  let up = h * sqrt(1.0 - lean * lean) * (1.0 - 0.45 * press);
  let w = size.y * pow(1.0 - min(t, 1.0), 0.9) * select(1.0, 0.0, lv.x >= 3.0);
  let world = root + right * (lv.y * w * 0.5) + vec3f(bend * (t * t), up * t);
  out.pos = to_clip(world);
  let tip = unpack(color);
  let shade = 0.88 + 0.24 * hash(seed + 13u);
  // Darker at the root (the blades shade each other), light coming through near the tip, a
  // highlight where the wind turns the blade to the light; a few blades carry a flower.
  var col = mix(tip * 0.32, tip * 1.08, t * t) * shade;
  col = col + vec3f(0.08, 0.12, 0.02) * smoothstep(0.5, 1.0, t) * (0.5 + 0.5 * gust);
  let flower = hash(seed + 29u);
  if (flower < 0.035 && lv.x >= 2.0) {
    let fc = select(select(vec3f(0.95, 0.92, 0.85), vec3f(0.95, 0.55, 0.75), flower < 0.02), vec3f(0.98, 0.82, 0.25),
                    flower < 0.01);
    col = mix(col, fc, smoothstep(2.0, 3.0, lv.x));
  }
  out.color = col;
  return out;
}

// A rock: an octahedron with each corner pulled in or out by its seed, eight flat faces.
const CORNER = array<vec3f, 6>(vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0),
                               vec3f(0.0, -1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0));
const FACE = array<u32, 24>(0u, 2u, 4u, 2u, 1u, 4u, 1u, 3u, 4u, 3u, 0u, 4u,
                            2u, 0u, 5u, 1u, 2u, 5u, 3u, 1u, 5u, 0u, 3u, 5u);

fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

fn rock_corner(i: u32, seed: u32) -> vec3f {
  let c = CORNER[i];
  let r = 0.55 + 0.45 * hash(seed * 7u + i);
  return c * r * vec3f(1.0, 0.8 + 0.3 * hash(seed + 3u), 0.6 + 0.3 * hash(seed + 5u));
}

@vertex
fn vs_rock(@builtin(vertex_index) vi: u32, @location(0) posScale: vec4f, @location(1) rot: vec4f,
           @location(2) color: u32, @location(3) seed: u32) -> VOut {
  var out: VOut;
  let f = vi / 3u;
  let a = qrot(rot, rock_corner(FACE[f * 3u], seed));
  let b = qrot(rot, rock_corner(FACE[f * 3u + 1u], seed));
  let c = qrot(rot, rock_corner(FACE[f * 3u + 2u], seed));
  var n = normalize(cross(b - a, c - a));
  let corner = select(select(c, b, vi % 3u == 1u), a, vi % 3u == 0u);
  out.pos = to_clip(posScale.xyz + corner * posScale.w * 0.5);
  let light = normalize(vec3f(0.3, 0.4, 0.85));
  let lambert = max(dot(n, light), 0.0);
  out.color = unpack(color) * (0.45 + 0.7 * lambert) * (0.9 + 0.2 * hash(seed + f));
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  var c = in.color;
  // GX's perspective fog, as the world's draws have it.
  let ftype = u32(u.fog.w);
  if (ftype >= 2u && ftype < 8u) {
    let depth = REVERSED_DEPTH;
    let f = clamp(u.fog.x / (u.fog.y - depth) - u.fog.z, 0.0, 1.0);
    var z = f;
    if (ftype == 4u) { z = 1.0 - exp2(-8.0 * f); }
    else if (ftype == 5u) { z = 1.0 - exp2(-8.0 * f * f); }
    else if (ftype == 6u) { z = exp2(-8.0 * (1.0 - f)); }
    else if (ftype == 7u) { let g = 1.0 - f; z = exp2(-8.0 * g * g); }
    c = mix(c, u.fogColor.rgb, clamp(z, 0.0, 1.0));
  }
  return vec4f(c, 1.0);
}
)";

struct State {
  DrawTypeId drawType = InvalidDrawType;
  wgpu::BindGroupLayout bindLayout;
  wgpu::PipelineLayout pipelineLayout;
  wgpu::ShaderModule module;
  std::map<std::array<uint64_t, 4>, wgpu::RenderPipeline> pipelines;
  std::map<const void*, wgpu::BindGroup> groups;
  // Render thread: the blades on the GPU.
  struct Grass {
    wgpu::Buffer buffer;
    uint32_t count = 0;
  };
  std::unordered_map<uint64_t, Grass> grass;
};
State g_state;

// Game thread -> render thread: blades to upload (an empty vector forgets the key).
std::mutex g_pendingMutex;
std::unordered_map<uint64_t, std::vector<Blade>> g_pending;

void ensure_static(const wgpu::Device& device) {
  if (g_state.module) {
    return;
  }
  const wgpu::BindGroupLayoutEntry entry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
      .buffer = {.type = wgpu::BufferBindingType::Uniform, .hasDynamicOffset = true, .minBindingSize = sizeof(Uniform)},
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "Geo Bind Group Layout",
      .entryCount = 1,
      .entries = &entry,
  };
  g_state.bindLayout = device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "Geo Pipeline Layout",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_state.bindLayout,
  };
  g_state.pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  std::string source = ShaderSource;
  const std::string token = "REVERSED_DEPTH";
  source.replace(source.find(token), token.size(), gx::UseReversedZ ? "(1.0 - in.pos.z)" : "in.pos.z");
  wgpu::ShaderSourceWGSL wgsl{};
  wgsl.code = source.c_str();
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "Geo Module"};
  g_state.module = device.CreateShaderModule(&moduleDescriptor);
}

wgpu::RenderPipeline pipeline_for(const DrawContext& ctx, uint32_t kind) {
  const auto& layout = ctx.layout;
  const std::array<uint64_t, 4> key{layout.key, layout.sampleCount, kind,
                                    static_cast<uint64_t>(layout.depthStencilFormat)};
  const auto it = g_state.pipelines.find(key);
  if (it != g_state.pipelines.end()) {
    return it->second;
  }
  const bool grass = kind == KindGrass;
  const std::array<wgpu::VertexAttribute, 4> grassAttributes{{
      {.format = wgpu::VertexFormat::Float32x4, .offset = 0, .shaderLocation = 0},
      {.format = wgpu::VertexFormat::Float32x2, .offset = 16, .shaderLocation = 1},
      {.format = wgpu::VertexFormat::Uint32, .offset = 24, .shaderLocation = 2},
      {.format = wgpu::VertexFormat::Uint32, .offset = 28, .shaderLocation = 3},
  }};
  const std::array<wgpu::VertexAttribute, 4> rockAttributes{{
      {.format = wgpu::VertexFormat::Float32x4, .offset = 0, .shaderLocation = 0},
      {.format = wgpu::VertexFormat::Float32x4, .offset = 16, .shaderLocation = 1},
      {.format = wgpu::VertexFormat::Uint32, .offset = 32, .shaderLocation = 2},
      {.format = wgpu::VertexFormat::Uint32, .offset = 36, .shaderLocation = 3},
  }};
  const wgpu::VertexBufferLayout vertexLayout{
      .stepMode = wgpu::VertexStepMode::Instance,
      .arrayStride = grass ? sizeof(Blade) : sizeof(Rock),
      .attributeCount = 4,
      .attributes = grass ? grassAttributes.data() : rockAttributes.data(),
  };
  std::array<wgpu::ColorTargetState, MaxColorAttachments> targets{};
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    targets[i] = {
        .format = layout.colorAttachments[i].format,
        .writeMask = i == SceneColorAttachmentIndex ? wgpu::ColorWriteMask::All : wgpu::ColorWriteMask::None,
    };
  }
  const wgpu::FragmentState fragment{
      .module = g_state.module,
      .entryPoint = "fs_main",
      .targetCount = layout.colorAttachmentCount,
      .targets = targets.data(),
  };
  const wgpu::DepthStencilState depth{
      .format = layout.depthStencilFormat,
      .depthWriteEnabled = true,
      .depthCompare = gx::UseReversedZ ? wgpu::CompareFunction::GreaterEqual : wgpu::CompareFunction::LessEqual,
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = grass ? "Geo Grass" : "Geo Rocks",
      .layout = g_state.pipelineLayout,
      .vertex = {.module = g_state.module,
                 .entryPoint = grass ? "vs_grass" : "vs_rock",
                 .bufferCount = 1,
                 .buffers = &vertexLayout},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList, .cullMode = wgpu::CullMode::None},
      .depthStencil = layout.depthStencilFormat != wgpu::TextureFormat::Undefined ? &depth : nullptr,
      .multisample = {.count = layout.sampleCount, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  auto pipeline = ctx.device.CreateRenderPipeline(&descriptor);
  g_state.pipelines.emplace(key, pipeline);
  return pipeline;
}

void upload_pending(const DrawContext& ctx) {
  std::unordered_map<uint64_t, std::vector<Blade>> pending;
  {
    std::scoped_lock lock{g_pendingMutex};
    if (g_pending.empty()) {
      return;
    }
    pending.swap(g_pending);
  }
  for (auto& [key, blades] : pending) {
    if (blades.empty()) {
      g_state.grass.erase(key);
      continue;
    }
    const wgpu::BufferDescriptor descriptor{
        .label = "Geo Grass Blades",
        .usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst,
        .size = blades.size() * sizeof(Blade),
    };
    State::Grass g;
    g.buffer = ctx.device.CreateBuffer(&descriptor);
    ctx.queue.WriteBuffer(g.buffer, 0, blades.data(), blades.size() * sizeof(Blade));
    g.count = static_cast<uint32_t>(blades.size());
    g_state.grass[key] = std::move(g);
    Log.info("grass {:016x}: {} blades", key, blades.size());
  }
}

void draw_payload(const DrawContext& ctx, const wgpu::RenderPassEncoder& pass, const void* data, size_t size, void*) {
  if (size != sizeof(Payload)) {
    return;
  }
  Payload p;
  std::memcpy(&p, data, sizeof(p));
  ensure_static(ctx.device);
  upload_pending(ctx);
  wgpu::Buffer instances;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  uint32_t count = p.count;
  if (p.kind == KindGrass) {
    const auto it = g_state.grass.find(p.key);
    if (it == g_state.grass.end() || p.first >= it->second.count) {
      return;
    }
    count = std::min(count, it->second.count - p.first);
    instances = it->second.buffer;
    offset = static_cast<uint64_t>(p.first) * sizeof(Blade);
    bytes = static_cast<uint64_t>(count) * sizeof(Blade);
  } else {
    instances = ctx.vertexBuffer;
    offset = p.instances.offset;
    bytes = p.instances.size;
  }
  if (count == 0 || !instances) {
    return;
  }
  auto group = g_state.groups.find(ctx.uniformBuffer.Get());
  if (group == g_state.groups.end()) {
    const wgpu::BindGroupEntry entry{.binding = 0, .buffer = ctx.uniformBuffer, .size = sizeof(Uniform)};
    const wgpu::BindGroupDescriptor descriptor{
        .label = "Geo Bind Group",
        .layout = g_state.bindLayout,
        .entryCount = 1,
        .entries = &entry,
    };
    group = g_state.groups.emplace(ctx.uniformBuffer.Get(), ctx.device.CreateBindGroup(&descriptor)).first;
  }
  pass.SetPipeline(pipeline_for(ctx, p.kind));
  const uint32_t dynamicOffset = p.uniform.offset;
  pass.SetBindGroup(0, group->second, 1, &dynamicOffset);
  pass.SetVertexBuffer(0, instances, offset, bytes);
  pass.Draw(p.kind == KindGrass ? 15 : 24, count);
}

bool ensure_type() {
  if (g_state.drawType == InvalidDrawType) {
    g_state.drawType = register_draw_type(DrawTypeDescriptor{.label = "Geo", .draw = draw_payload});
  }
  return g_state.drawType != InvalidDrawType;
}

Range push_frame_uniform(const GrassDraw* grass) {
  const auto& gx = gx::g_gxState;
  Uniform u{};
  Mat4x4<float> proj = gx.proj;
  if (gx::UseReversedZ) {
    proj.m2 = proj.m2 * Vec4<float>{-1.f, -1.f, -1.f, -1.f};
  } else {
    proj.m2 = proj.m2 + proj.m3;
  }
  std::memcpy(u.proj, &proj, sizeof(u.proj));
  std::memcpy(u.mv, &gx.pnMtx[gx.currentPnMtx].pos, sizeof(u.mv));
  if (grass != nullptr) {
    u.params[0] = grass->time;
    u.params[1] = grass->wind[0];
    u.params[2] = grass->wind[1];
    u.params[3] = grass->windStrength;
    u.fade[0] = grass->fadeStart;
    u.fade[1] = grass->fadeEnd;
    const uint32_t pushers = std::min(grass->pusherCount, GrassDraw::MaxPushers);
    u.fade[2] = static_cast<float>(pushers);
    std::memcpy(u.pushers, grass->pushers, sizeof(float) * 4 * pushers);
  }
  u.fogColor[0] = gx.fog.color.x();
  u.fogColor[1] = gx.fog.color.y();
  u.fogColor[2] = gx.fog.color.z();
  u.fog[0] = gx.fog.a;
  u.fog[1] = gx.fog.b;
  u.fog[2] = gx.fog.c;
  u.fog[3] = static_cast<float>(gx.fog.type);
  return push_uniform(reinterpret_cast<const uint8_t*>(&u), sizeof(u));
}
} // namespace

void set_grass(uint64_t key, const Blade* blades, uint32_t count) {
  std::scoped_lock lock{g_pendingMutex};
  auto& v = g_pending[key];
  if (blades == nullptr || count == 0) {
    v.clear();
  } else {
    v.assign(blades, blades + count);
  }
}

void draw_grass(const GrassDraw& draw) {
  if (draw.count == 0 || !ensure_type()) {
    return;
  }
  // The recorder and the GX state belong to the FIFO thread: let it catch up first (the draw then
  // lands after everything recorded so far, with the matrices just set).
  gx::fifo::drain();
  Payload p{};
  p.kind = KindGrass;
  p.count = draw.count;
  p.first = draw.first;
  p.key = draw.key;
  p.uniform = push_frame_uniform(&draw);
  if (overflowed(p.uniform) || p.uniform.size == 0) {
    return;
  }
  push_custom_draw(g_state.drawType, &p, sizeof(p));
}

void draw_rocks(const Rock* rocks, uint32_t count) {
  if (rocks == nullptr || count == 0 || !ensure_type()) {
    return;
  }
  gx::fifo::drain();
  Payload p{};
  p.kind = KindRocks;
  p.count = count;
  p.uniform = push_frame_uniform(nullptr);
  p.instances = push_verts(reinterpret_cast<const uint8_t*>(rocks), size_t(count) * sizeof(Rock), 4);
  if (overflowed(p.uniform) || overflowed(p.instances) || p.uniform.size == 0 || p.instances.size == 0) {
    return;
  }
  push_custom_draw(g_state.drawType, &p, sizeof(p));
}

void shutdown() {
  const auto type = g_state.drawType;
  g_state = {};
  {
    std::scoped_lock lock{g_pendingMutex};
    g_pending.clear();
  }
  if (type != InvalidDrawType) {
    unregister_draw_type(type);
  }
}
} // namespace aurora::gfx::geo
