#include "ssao.hpp"

#include "../logging.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "recording.hpp"

#include <aurora/gfx.hpp>

#include <array>
#include <cstring>
#include <string>

// Screen-space ambient occlusion (see ssao.hpp). The depth buffer is read as volfog.cpp reads it:
// reversed Z, the world drawn in GX z depth[2]..depth[3] and the viewmodel nearer than that.
namespace aurora::gfx::ssao {
namespace {
Module Log("aurora::gfx::ssao");
using webgpu::g_device;

constexpr auto AoFormat = wgpu::TextureFormat::R8Unorm;

constexpr const char* CommonSource = R"(
struct Params {
  frustum: vec4f,
  depth: vec4f,
  ao: vec4f,
  misc: vec4u,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var depthTex: DEPTH_TYPE;

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

// The world's view depth (positive, in world units) at a depth texel; 0 for the viewmodel and
// the sky, which take no occlusion and give none.
fn linear_depth(at: vec2i) -> f32 {
  let size = vec2i(textureDimensions(depthTex));
  let z = 1.0 - textureLoad(depthTex, clamp(at, vec2i(0), size - vec2i(1)), 0);
  if (z < p.depth.z || z >= 0.99999) {
    return 0.0;
  }
  let d = clamp((z - p.depth.z) / max(p.depth.w - p.depth.z, 1e-6), 0.0, 1.0);
  let near = p.depth.x;
  let far = p.depth.y;
  return near * far / (far - d * (far - near));
}

// The view-space position (GX's: x right, y up, z towards the camera) at a uv and view depth.
fn view_pos(uv: vec2f, zl: f32) -> vec3f {
  let ray = vec2f(mix(p.frustum.x, p.frustum.y, uv.x), mix(p.frustum.z, p.frustum.w, 1.0 - uv.y));
  return vec3f(ray * zl, -zl);
}

// And back: the uv a view-space point projects to.
fn project(v: vec3f) -> vec2f {
  let zl = max(-v.z, 1e-4);
  let ray = v.xy / zl;
  let s = (ray - vec2f(p.frustum.x, p.frustum.z)) / vec2f(p.frustum.y - p.frustum.x, p.frustum.w - p.frustum.z);
  return vec2f(s.x, 1.0 - s.y);
}
)";

constexpr const char* AoSource = R"(
fn ign(px: vec2f, frame: u32) -> f32 {
  let q = px + 5.588238 * f32(frame % 64u);
  return fract(52.9829189 * fract(dot(q, vec2f(0.06711056, 0.00583715))));
}

@fragment
fn fs_ao(in: VertexOutput) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(depthTex));
  let at = min(vec2i(in.uv * vec2f(size)), size - vec2i(1));
  let zl = linear_depth(at);
  if (zl <= 0.0 || zl > p.ao.w) {
    return vec4f(1.0);
  }
  let texel = 1.0 / vec2f(size);
  let uv = (vec2f(at) + 0.5) * texel;
  let pos = view_pos(uv, zl);
  // The normal from the nearer neighbour on each axis, so silhouettes don't bend it.
  let zr = linear_depth(at + vec2i(1, 0));
  let zlf = linear_depth(at - vec2i(1, 0));
  let zd = linear_depth(at + vec2i(0, 1));
  let zu = linear_depth(at - vec2i(0, 1));
  var dx: vec3f;
  if (zr > 0.0 && (zlf <= 0.0 || abs(zr - zl) < abs(zlf - zl))) {
    dx = view_pos(uv + vec2f(texel.x, 0.0), zr) - pos;
  } else if (zlf > 0.0) {
    dx = pos - view_pos(uv - vec2f(texel.x, 0.0), zlf);
  } else {
    return vec4f(1.0);
  }
  var dy: vec3f;
  if (zd > 0.0 && (zu <= 0.0 || abs(zd - zl) < abs(zu - zl))) {
    dy = view_pos(uv + vec2f(0.0, texel.y), zd) - pos;
  } else if (zu > 0.0) {
    dy = pos - view_pos(uv - vec2f(0.0, texel.y), zu);
  } else {
    return vec4f(1.0);
  }
  var n = normalize(cross(dy, dx));
  if (dot(n, -pos) < 0.0) {
    n = -n;
  }
  // A basis around the normal, turned per pixel.
  let angle = ign(in.pos.xy, p.misc.y) * 6.2831853;
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.99) {
    up = vec3f(1.0, 0.0, 0.0);
  }
  let t0 = normalize(cross(up, n));
  let b0 = cross(n, t0);
  let t = t0 * cos(angle) + b0 * sin(angle);
  let b = cross(n, t);

  let radius = p.ao.x;
  let bias = p.ao.z;
  let count = clamp(p.misc.x, 4u, 32u);
  var occlusion = 0.0;
  for (var i = 0u; i < count; i++) {
    let fi = f32(i);
    let u = (fi + 0.5) / f32(count);
    // Cosine-weighted directions on a golden-angle spiral, denser near the pixel.
    let phi = fi * 2.3999632;
    let sinT = sqrt(u);
    let cosT = sqrt(1.0 - u);
    let dir = t * (cos(phi) * sinT) + b * (sin(phi) * sinT) + n * cosT;
    let k = (fi + 1.0) / f32(count);
    let s = pos + dir * (radius * mix(0.1, 1.0, k * k));
    let suv = project(s);
    if (any(suv < vec2f(0.0)) || any(suv > vec2f(1.0))) {
      continue;
    }
    let sz = linear_depth(vec2i(suv * vec2f(size)));
    if (sz <= 0.0) {
      continue;
    }
    // In front of the sample: occluding, less so the farther that surface is from the pixel.
    let sceneZ = -sz;
    let range = smoothstep(0.0, 1.0, radius / max(abs(pos.z - sceneZ), 1e-4));
    occlusion += select(0.0, 1.0, sceneZ >= s.z + bias) * range;
  }
  let fade = 1.0 - smoothstep(p.ao.w * 0.6, p.ao.w, zl);
  let ao = 1.0 - occlusion / f32(count) * fade;
  return vec4f(ao, ao, ao, 1.0);
}
)";

constexpr const char* ApplySource = R"(
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var aoTex: texture_2d<f32>;

fn srgb_enc(c: vec3f) -> vec3f {
  let l = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(l, vec3f(1.0 / 2.4)) - 0.055, 12.92 * l, l <= vec3f(0.0031308));
}

fn srgb_dec(c: vec3f) -> vec3f {
  let e = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(pow((e + 0.055) / 1.055, vec3f(2.4)), e / 12.92, e <= vec3f(0.04045));
}

@fragment
fn fs_apply(in: VertexOutput) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(src));
  let px = min(vec2i(floor(in.pos.xy)), size - vec2i(1));
  let f = textureLoad(src, px, 0);
  let aoSize = vec2i(textureDimensions(aoTex));
  let at = min(vec2i(in.uv * vec2f(aoSize)), aoSize - vec2i(1));
  let zl = linear_depth(at);
  if (zl <= 0.0) {
    return f;
  }
  // A 5x5 blur of the occlusion over texels at about the same depth.
  var sum = 0.0;
  var weight = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let o = clamp(at + vec2i(x, y), vec2i(0), aoSize - vec2i(1));
      let oz = linear_depth(o);
      let w = select(0.0, 1.0 / (1.0 + abs(oz - zl) * 8.0 / max(zl, 1e-3)), oz > 0.0);
      sum += textureLoad(aoTex, o, 0).r * w;
      weight += w;
    }
  }
  let ao = select(1.0, sum / weight, weight > 0.0);
  let k = mix(1.0, ao, clamp(p.ao.y, 0.0, 1.0));
  if (k >= 0.999) {
    return f;
  }
  return vec4f(srgb_enc(srgb_dec(f.rgb) * k), f.a);
}
)";

struct State {
  EncoderTaskId task = InvalidEncoderTask;
  wgpu::Buffer uniforms;
  wgpu::BindGroupLayout aoLayout;
  wgpu::RenderPipeline ao;
  wgpu::BindGroupLayout applyLayout;
  wgpu::RenderPipeline apply;
  wgpu::TextureFormat applyFormat = wgpu::TextureFormat::Undefined;
  uint32_t samples = 0;
  wgpu::Texture aoTexture;
  wgpu::TextureView aoView;
  wgpu::Texture frame;
  wgpu::TextureView frameView;
  wgpu::TextureFormat frameFormat = wgpu::TextureFormat::Undefined;
  uint32_t width = 0;
  uint32_t height = 0;
};
State g_state;

wgpu::ShaderModule make_module(const char* label, std::string code, uint32_t samples) {
  const std::string token = "DEPTH_TYPE";
  code.replace(code.find(token), token.size(), samples > 1 ? "texture_depth_multisampled_2d" : "texture_depth_2d");
  wgpu::ShaderSourceWGSL source{};
  source.code = code.c_str();
  const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &source, .label = label};
  return g_device.CreateShaderModule(&descriptor);
}

wgpu::BindGroupLayoutEntry uniform_entry() {
  return wgpu::BindGroupLayoutEntry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Fragment,
      .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(Params)},
  };
}

wgpu::BindGroupLayoutEntry depth_entry(uint32_t samples) {
  return wgpu::BindGroupLayoutEntry{
      .binding = 1,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture = {.sampleType = wgpu::TextureSampleType::Depth,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
                  .multisampled = samples > 1},
  };
}

wgpu::BindGroupLayoutEntry float_entry(uint32_t binding) {
  return wgpu::BindGroupLayoutEntry{
      .binding = binding,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                  .viewDimension = wgpu::TextureViewDimension::e2D},
  };
}

wgpu::RenderPipeline make_pipeline(const char* label, const wgpu::ShaderModule& module, const char* entry,
                                   const wgpu::BindGroupLayout& layout, wgpu::TextureFormat format,
                                   uint32_t samples) {
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = label,
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &layout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  const wgpu::ColorTargetState target{.format = format, .writeMask = wgpu::ColorWriteMask::All};
  const wgpu::FragmentState fragment{
      .module = module,
      .entryPoint = entry,
      .targetCount = 1,
      .targets = &target,
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = label,
      .layout = pipelineLayout,
      .vertex = {.module = module, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

void ensure_pipelines(wgpu::TextureFormat format, uint32_t samples) {
  if (!g_state.uniforms) {
    const wgpu::BufferDescriptor bufferDescriptor{
        .label = "SSAO Uniforms",
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(Params),
    };
    g_state.uniforms = g_device.CreateBuffer(&bufferDescriptor);
  }
  if (g_state.ao && g_state.apply && g_state.applyFormat == format && g_state.samples == samples) {
    return;
  }
  {
    const std::array entries{uniform_entry(), depth_entry(samples)};
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
        .label = "SSAO Layout",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_state.aoLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
    const auto module = make_module("SSAO", std::string(CommonSource) + AoSource, samples);
    // The occlusion has one sample a pixel, whatever the frame's count.
    g_state.ao = make_pipeline("SSAO", module, "fs_ao", g_state.aoLayout, AoFormat, 1);
  }
  {
    const std::array entries{uniform_entry(), depth_entry(samples), float_entry(2), float_entry(3)};
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
        .label = "SSAO Apply Layout",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_state.applyLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
    const auto module = make_module("SSAO Apply", std::string(CommonSource) + ApplySource, samples);
    g_state.apply = make_pipeline("SSAO Apply", module, "fs_apply", g_state.applyLayout, format, samples);
  }
  g_state.applyFormat = format;
  g_state.samples = samples;
}

void ensure_targets(uint32_t width, uint32_t height, wgpu::TextureFormat format) {
  if (g_state.frame && g_state.width == width && g_state.height == height && g_state.frameFormat == format) {
    return;
  }
  g_state.width = width;
  g_state.height = height;
  g_state.frameFormat = format;
  const wgpu::TextureDescriptor frameDescriptor{
      .label = "SSAO Frame Copy",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .size = {width, height, 1},
      .format = format,
  };
  g_state.frame = g_device.CreateTexture(&frameDescriptor);
  g_state.frameView = g_state.frame.CreateView();
  const wgpu::TextureDescriptor aoDescriptor{
      .label = "SSAO Occlusion",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment,
      .size = {width, height, 1},
      .format = AoFormat,
  };
  g_state.aoTexture = g_device.CreateTexture(&aoDescriptor);
  g_state.aoView = g_state.aoTexture.CreateView();
}

void encode(const EncoderTaskContext& ctx, const wgpu::CommandEncoder& cmd, const void* payload, size_t payloadSize,
            void*) {
  if (payloadSize != sizeof(Params)) {
    return;
  }
  Params params;
  std::memcpy(&params, payload, sizeof(params));
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
  ensure_targets(width, height, format);
  ctx.queue.WriteBuffer(g_state.uniforms, 0, &params, sizeof(params));

  {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .buffer = g_state.uniforms, .size = sizeof(Params)},
        wgpu::BindGroupEntry{.binding = 1, .textureView = depth.view},
    };
    const wgpu::BindGroupDescriptor groupDescriptor{
        .label = "SSAO",
        .layout = g_state.aoLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    const auto group = g_device.CreateBindGroup(&groupDescriptor);
    const wgpu::RenderPassColorAttachment attachment{
        .view = g_state.aoView,
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {1.0, 1.0, 1.0, 1.0},
    };
    const wgpu::RenderPassDescriptor passDescriptor{
        .label = "SSAO",
        .colorAttachmentCount = 1,
        .colorAttachments = &attachment,
        .timestampWrites = webgpu::gpu_prof::pass_writes("SSAO"),
    };
    const auto pass = cmd.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(g_state.ao);
    pass.SetBindGroup(0, group);
    pass.Draw(3);
    pass.End();
  }

  const wgpu::TexelCopyTextureInfo copySource{.texture = source.texture};
  const wgpu::TexelCopyTextureInfo copyTarget{.texture = g_state.frame};
  const wgpu::Extent3D copySize{width, height, 1};
  cmd.CopyTextureToTexture(&copySource, &copyTarget, &copySize);

  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = g_state.uniforms, .size = sizeof(Params)},
      wgpu::BindGroupEntry{.binding = 1, .textureView = depth.view},
      wgpu::BindGroupEntry{.binding = 2, .textureView = g_state.frameView},
      wgpu::BindGroupEntry{.binding = 3, .textureView = g_state.aoView},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "SSAO Apply",
      .layout = g_state.applyLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto group = g_device.CreateBindGroup(&groupDescriptor);
  // Every pixel is written without blending, so the frame need not be loaded first.
  const wgpu::RenderPassColorAttachment attachment{
      .view = target.view,
      .resolveTarget = samples > 1 ? webgpu::g_frameBufferResolved.view : wgpu::TextureView{},
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
  };
  const wgpu::RenderPassDescriptor passDescriptor{
      .label = "SSAO Apply",
      .colorAttachmentCount = 1,
      .colorAttachments = &attachment,
      .timestampWrites = webgpu::gpu_prof::pass_writes("SSAO apply"),
  };
  const auto pass = cmd.BeginRenderPass(&passDescriptor);
  pass.SetPipeline(g_state.apply);
  pass.SetBindGroup(0, group);
  pass.Draw(3);
  pass.End();
}
} // namespace

bool ensure_task() {
  if (g_state.task == InvalidEncoderTask) {
    g_state.task = register_encoder_task_type(EncoderTaskDescriptor{.label = "SSAO", .callback = encode});
    if (g_state.task == InvalidEncoderTask) {
      Log.warn("could not register the SSAO task");
      return false;
    }
  }
  return true;
}

bool record(const Params& params) {
  if (g_state.task == InvalidEncoderTask) {
    return false;
  }
  const auto& size = webgpu::present_source().size;
  if (size.width == 0 || size.height == 0) {
    return false;
  }
  return record_encoder_task(g_state.task, &params, sizeof(params));
}

void shutdown() {
  const auto task = g_state.task;
  g_state = {};
  if (task != InvalidEncoderTask) {
    unregister_encoder_task_type(task);
  }
}
} // namespace aurora::gfx::ssao
