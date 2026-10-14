#include "ssgi.hpp"

#include "../logging.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "recording.hpp"

#include <aurora/gfx.hpp>

#include <array>
#include <cstring>
#include <string>

// Screen-space global illumination (see ssgi.hpp): a compute pass traces, a render pass applies.
// The depth buffer is read as ssao.cpp reads it.
namespace aurora::gfx::ssgi {
namespace {
Module Log("aurora::gfx::ssgi");
using webgpu::g_device;

constexpr auto GiFormat = wgpu::TextureFormat::RGBA16Float;
constexpr uint32_t Group = 8;

constexpr const char* CommonSource = R"(
struct Params {
  frustum: vec4f,
  depth: vec4f,
  gi: vec4f,
  misc: vec4u,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var depthTex: DEPTH_TYPE;

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

fn view_pos(uv: vec2f, zl: f32) -> vec3f {
  let ray = vec2f(mix(p.frustum.x, p.frustum.y, uv.x), mix(p.frustum.z, p.frustum.w, 1.0 - uv.y));
  return vec3f(ray * zl, -zl);
}

fn project(v: vec3f) -> vec2f {
  let zl = max(-v.z, 1e-4);
  let ray = v.xy / zl;
  let s = (ray - vec2f(p.frustum.x, p.frustum.z)) / vec2f(p.frustum.y - p.frustum.x, p.frustum.w - p.frustum.z);
  return vec2f(s.x, 1.0 - s.y);
}

fn srgb_enc(c: vec3f) -> vec3f {
  let l = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(l, vec3f(1.0 / 2.4)) - 0.055, 12.92 * l, l <= vec3f(0.0031308));
}

fn srgb_dec(c: vec3f) -> vec3f {
  let e = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(pow((e + 0.055) / 1.055, vec3f(2.4)), e / 12.92, e <= vec3f(0.04045));
}
)";

constexpr const char* TraceSource = R"(
@group(0) @binding(2) var colorTex: texture_2d<f32>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba16float, write>;

fn ign(px: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8)
fn cs_trace(@builtin(global_invocation_id) gid: vec3u) {
  let osize = textureDimensions(outTex);
  if (gid.x >= osize.x || gid.y >= osize.y) {
    return;
  }
  let size = vec2i(textureDimensions(depthTex));
  let at = min(vec2i((vec2f(gid.xy) + 0.5) * vec2f(size) / vec2f(osize)), size - vec2i(1));
  let zl = linear_depth(at);
  if (zl <= 0.0 || zl > p.gi.w) {
    textureStore(outTex, gid.xy, vec4f(0.0));
    return;
  }
  let texel = 1.0 / vec2f(size);
  let uv = (vec2f(at) + 0.5) * texel;
  let pos = view_pos(uv, zl);
  let zr = linear_depth(at + vec2i(1, 0));
  let zlf = linear_depth(at - vec2i(1, 0));
  let zd = linear_depth(at + vec2i(0, 1));
  let zu = linear_depth(at - vec2i(0, 1));
  var dx = vec3f(0.0);
  if (zr > 0.0 && (zlf <= 0.0 || abs(zr - zl) < abs(zlf - zl))) {
    dx = view_pos(uv + vec2f(texel.x, 0.0), zr) - pos;
  } else if (zlf > 0.0) {
    dx = pos - view_pos(uv - vec2f(texel.x, 0.0), zlf);
  }
  var dy = vec3f(0.0);
  if (zd > 0.0 && (zu <= 0.0 || abs(zd - zl) < abs(zu - zl))) {
    dy = view_pos(uv + vec2f(0.0, texel.y), zd) - pos;
  } else if (zu > 0.0) {
    dy = pos - view_pos(uv - vec2f(0.0, texel.y), zu);
  }
  if (dot(dx, dx) == 0.0 || dot(dy, dy) == 0.0) {
    textureStore(outTex, gid.xy, vec4f(0.0));
    return;
  }
  var n = normalize(cross(dy, dx));
  if (dot(n, -pos) < 0.0) {
    n = -n;
  }
  let angle = ign(vec2f(gid.xy)) * 6.2831853;
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.99) {
    up = vec3f(1.0, 0.0, 0.0);
  }
  let t0 = normalize(cross(up, n));
  let b0 = cross(n, t0);
  let t = t0 * cos(angle) + b0 * sin(angle);
  let b = cross(n, t);

  let rays = clamp(p.misc.x, 2u, 16u);
  let steps = clamp(p.misc.y, 4u, 24u);
  let len = p.gi.x;
  let thick = p.gi.z;
  let jitter = ign(vec2f(gid.yx) + 17.0);
  var acc = vec3f(0.0);
  for (var r = 0u; r < rays; r++) {
    let fr = f32(r);
    let u = (fr + 0.5) / f32(rays);
    let phi = fr * 2.3999632;
    let sinT = sqrt(u);
    let cosT = sqrt(1.0 - u);
    let dir = t * (cos(phi) * sinT) + b * (sin(phi) * sinT) + n * cosT;
    for (var s = 0u; s < steps; s++) {
      let k = (f32(s) + jitter) / f32(steps);
      let dist = len * (0.04 + 0.96 * k * k);
      let q = pos + dir * dist;
      let quv = project(q);
      if (any(quv < vec2f(0.0)) || any(quv > vec2f(1.0))) {
        break;
      }
      let qpx = vec2i(quv * vec2f(size));
      let qz = linear_depth(qpx);
      if (qz <= 0.0) {
        continue;
      }
      let behind = -q.z - qz;
      if (behind > 0.02 && behind < thick) {
        // A surface: the light it shows, nearer counts more.
        let c = srgb_dec(textureLoad(colorTex, min(qpx, vec2i(textureDimensions(colorTex)) - vec2i(1)), 0).rgb);
        acc += c * (1.0 - k * 0.7);
        break;
      }
    }
  }
  let fade = 1.0 - smoothstep(p.gi.w * 0.6, p.gi.w, zl);
  textureStore(outTex, gid.xy, vec4f(acc / f32(rays) * fade, 1.0));
}
)";

constexpr const char* ApplySource = R"(
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var giTex: texture_2d<f32>;

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

@fragment
fn fs_apply(in: VertexOutput) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(src));
  let px = min(vec2i(floor(in.pos.xy)), size - vec2i(1));
  let f = textureLoad(src, px, 0);
  let dsize = vec2i(textureDimensions(depthTex));
  let dat = min(vec2i(in.uv * vec2f(dsize)), dsize - vec2i(1));
  let zl = linear_depth(dat);
  if (zl <= 0.0) {
    return f;
  }
  // A 7x7 depth-aware blur of the half-size bounce light (the rays are few and noisy).
  let gsize = vec2i(textureDimensions(giTex));
  let gat = min(vec2i(in.uv * vec2f(gsize)), gsize - vec2i(1));
  var sum = vec3f(0.0);
  var weight = 0.0;
  for (var y = -3; y <= 3; y++) {
    for (var x = -3; x <= 3; x++) {
      let o = clamp(gat + vec2i(x, y), vec2i(0), gsize - vec2i(1));
      let ouv = (vec2f(o) + 0.5) / vec2f(gsize);
      let oz = linear_depth(min(vec2i(ouv * vec2f(dsize)), dsize - vec2i(1)));
      let w = select(0.0, exp(-f32(x * x + y * y) * 0.08) / (1.0 + abs(oz - zl) * 12.0 / max(zl, 1e-3)), oz > 0.0);
      sum += textureLoad(giTex, o, 0).rgb * w;
      weight += w;
    }
  }
  if (weight <= 0.0) {
    return f;
  }
  let gi = sum / weight;
  let c = srgb_dec(f.rgb);
  // The pixel's albedo, as far as its colour tells: its hue, at a brightness that doesn't
  // fall to nothing where the room is dark (that's where the bounce matters most).
  let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  let albedo = clamp(c / max(lum, 0.04) * 0.45, vec3f(0.0), vec3f(1.0));
  return vec4f(srgb_enc(c + gi * albedo * p.gi.y), f.a);
}
)";

struct State {
  EncoderTaskId task = InvalidEncoderTask;
  wgpu::Buffer uniforms;
  wgpu::BindGroupLayout traceLayout;
  wgpu::ComputePipeline trace;
  wgpu::BindGroupLayout applyLayout;
  wgpu::RenderPipeline apply;
  wgpu::TextureFormat applyFormat = wgpu::TextureFormat::Undefined;
  uint32_t samples = 0;
  wgpu::Texture giTexture;
  wgpu::TextureView giView;
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

wgpu::BindGroupLayoutEntry uniform_entry(wgpu::ShaderStage stage) {
  return wgpu::BindGroupLayoutEntry{
      .binding = 0,
      .visibility = stage,
      .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(Params)},
  };
}

wgpu::BindGroupLayoutEntry depth_entry(wgpu::ShaderStage stage, uint32_t samples) {
  return wgpu::BindGroupLayoutEntry{
      .binding = 1,
      .visibility = stage,
      .texture = {.sampleType = wgpu::TextureSampleType::Depth,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
                  .multisampled = samples > 1},
  };
}

wgpu::BindGroupLayoutEntry float_entry(wgpu::ShaderStage stage, uint32_t binding) {
  return wgpu::BindGroupLayoutEntry{
      .binding = binding,
      .visibility = stage,
      .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                  .viewDimension = wgpu::TextureViewDimension::e2D},
  };
}

void ensure_pipelines(wgpu::TextureFormat format, uint32_t samples) {
  if (!g_state.uniforms) {
    const wgpu::BufferDescriptor bufferDescriptor{
        .label = "SSGI Uniforms",
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(Params),
    };
    g_state.uniforms = g_device.CreateBuffer(&bufferDescriptor);
  }
  if (g_state.trace && g_state.apply && g_state.applyFormat == format && g_state.samples == samples) {
    return;
  }
  {
    const std::array entries{
        uniform_entry(wgpu::ShaderStage::Compute),
        depth_entry(wgpu::ShaderStage::Compute, samples),
        float_entry(wgpu::ShaderStage::Compute, 2),
        wgpu::BindGroupLayoutEntry{
            .binding = 3,
            .visibility = wgpu::ShaderStage::Compute,
            .storageTexture = {.access = wgpu::StorageTextureAccess::WriteOnly,
                               .format = GiFormat,
                               .viewDimension = wgpu::TextureViewDimension::e2D},
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
        .label = "SSGI Trace Layout",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_state.traceLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
    const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
        .label = "SSGI Trace",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &g_state.traceLayout,
    };
    const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
    const auto module = make_module("SSGI Trace", std::string(CommonSource) + TraceSource, samples);
    const wgpu::ComputePipelineDescriptor descriptor{
        .label = "SSGI Trace",
        .layout = pipelineLayout,
        .compute = {.module = module, .entryPoint = "cs_trace"},
    };
    g_state.trace = g_device.CreateComputePipeline(&descriptor);
  }
  {
    const auto stage = wgpu::ShaderStage::Fragment;
    const std::array entries{uniform_entry(stage), depth_entry(stage, samples), float_entry(stage, 2),
                             float_entry(stage, 3)};
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
        .label = "SSGI Apply Layout",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_state.applyLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
    const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
        .label = "SSGI Apply",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &g_state.applyLayout,
    };
    const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
    const auto module = make_module("SSGI Apply", std::string(CommonSource) + ApplySource, samples);
    const wgpu::ColorTargetState target{.format = format, .writeMask = wgpu::ColorWriteMask::All};
    const wgpu::FragmentState fragment{
        .module = module,
        .entryPoint = "fs_apply",
        .targetCount = 1,
        .targets = &target,
    };
    const wgpu::RenderPipelineDescriptor descriptor{
        .label = "SSGI Apply",
        .layout = pipelineLayout,
        .vertex = {.module = module, .entryPoint = "vs_main"},
        .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
        .multisample = {.count = samples, .mask = UINT32_MAX},
        .fragment = &fragment,
    };
    g_state.apply = g_device.CreateRenderPipeline(&descriptor);
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
      .label = "SSGI Frame Copy",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .size = {width, height, 1},
      .format = format,
  };
  g_state.frame = g_device.CreateTexture(&frameDescriptor);
  g_state.frameView = g_state.frame.CreateView();
  const wgpu::TextureDescriptor giDescriptor{
      .label = "SSGI Bounce",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::StorageBinding,
      .size = {(width + 1) / 2, (height + 1) / 2, 1},
      .format = GiFormat,
  };
  g_state.giTexture = g_device.CreateTexture(&giDescriptor);
  g_state.giView = g_state.giTexture.CreateView();
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

  // The frame as lit so far: what the rays find, and what the apply pass adds to.
  const wgpu::TexelCopyTextureInfo copySource{.texture = source.texture};
  const wgpu::TexelCopyTextureInfo copyTarget{.texture = g_state.frame};
  const wgpu::Extent3D copySize{width, height, 1};
  cmd.CopyTextureToTexture(&copySource, &copyTarget, &copySize);

  {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .buffer = g_state.uniforms, .size = sizeof(Params)},
        wgpu::BindGroupEntry{.binding = 1, .textureView = depth.view},
        wgpu::BindGroupEntry{.binding = 2, .textureView = g_state.frameView},
        wgpu::BindGroupEntry{.binding = 3, .textureView = g_state.giView},
    };
    const wgpu::BindGroupDescriptor groupDescriptor{
        .label = "SSGI Trace",
        .layout = g_state.traceLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    const auto group = g_device.CreateBindGroup(&groupDescriptor);
    const wgpu::ComputePassDescriptor passDescriptor{.label = "SSGI Trace"};
    const auto pass = cmd.BeginComputePass(&passDescriptor);
    pass.SetPipeline(g_state.trace);
    pass.SetBindGroup(0, group);
    const uint32_t gw = (width + 1) / 2, gh = (height + 1) / 2;
    pass.DispatchWorkgroups((gw + Group - 1) / Group, (gh + Group - 1) / Group, 1);
    pass.End();
  }

  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = g_state.uniforms, .size = sizeof(Params)},
      wgpu::BindGroupEntry{.binding = 1, .textureView = depth.view},
      wgpu::BindGroupEntry{.binding = 2, .textureView = g_state.frameView},
      wgpu::BindGroupEntry{.binding = 3, .textureView = g_state.giView},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "SSGI Apply",
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
      .label = "SSGI Apply",
      .colorAttachmentCount = 1,
      .colorAttachments = &attachment,
      .timestampWrites = webgpu::gpu_prof::pass_writes("SSGI apply"),
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
    g_state.task = register_encoder_task_type(EncoderTaskDescriptor{.label = "SSGI", .callback = encode});
    if (g_state.task == InvalidEncoderTask) {
      Log.warn("could not register the SSGI task");
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
} // namespace aurora::gfx::ssgi
