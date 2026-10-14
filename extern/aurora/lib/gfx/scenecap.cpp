#include "scenecap.hpp"

#include "../logging.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "recording.hpp"

#include <aurora/gfx.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

// The opaque frame's copy (see scenecap.hpp). Depth is read as volfog.cpp reads it.
namespace aurora::gfx::scenecap {
namespace {
Module Log("aurora::gfx::scenecap");
using webgpu::g_device;

constexpr auto CopyFormat = wgpu::TextureFormat::RGBA16Float;

constexpr const char* ShaderSource = R"(
struct Params {
  depth: vec4f,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var depthTex: DEPTH_TYPE;

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
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  // The copy may be smaller than the frame (3D textures are capped at 2048 a side): read by uv.
  let size = vec2i(textureDimensions(src));
  let px = min(vec2i(in.uv * vec2f(size)), size - vec2i(1));
  let c = textureLoad(src, px, 0);
  let dsize = vec2i(textureDimensions(depthTex));
  let at = min(vec2i(in.uv * vec2f(dsize)), dsize - vec2i(1));
  let z = 1.0 - textureLoad(depthTex, at, 0);
  var zl = 1e4;
  if (z >= p.depth.z && z < 0.99999) {
    let d = clamp((z - p.depth.z) / max(p.depth.w - p.depth.z, 1e-6), 0.0, 1.0);
    zl = p.depth.x * p.depth.y / (p.depth.y - d * (p.depth.y - p.depth.x));
  }
  return vec4f(c.rgb, zl);
}
)";

struct State {
  EncoderTaskId task = InvalidEncoderTask;
  wgpu::Buffer uniforms;
  wgpu::BindGroupLayout layout;
  wgpu::RenderPipeline pipeline;
  uint32_t samples = 0;
  wgpu::Texture copy;
  wgpu::TextureView copyView;     // 3D, for the draws that read it
  wgpu::TextureView copyTarget;   // the same, as the pass's attachment
  wgpu::TextureView placeholder;
  uint32_t width = 0;
  uint32_t height = 0;
  wgpu::TextureFormat frameFormat = wgpu::TextureFormat::Undefined;
  wgpu::Texture frame;
  wgpu::TextureView frameView;
};
State g_state;

void ensure_pipeline(uint32_t samples) {
  if (!g_state.uniforms) {
    const wgpu::BufferDescriptor bufferDescriptor{
        .label = "Scene Capture Uniforms",
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(Params),
    };
    g_state.uniforms = g_device.CreateBuffer(&bufferDescriptor);
  }
  if (g_state.pipeline && g_state.samples == samples) {
    return;
  }
  std::string code = ShaderSource;
  const std::string token = "DEPTH_TYPE";
  code.replace(code.find(token), token.size(), samples > 1 ? "texture_depth_multisampled_2d" : "texture_depth_2d");
  wgpu::ShaderSourceWGSL source{};
  source.code = code.c_str();
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "Scene Capture"};
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);
  const std::array entries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(Params)},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::Depth,
                      .viewDimension = wgpu::TextureViewDimension::e2D,
                      .multisampled = samples > 1},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "Scene Capture Layout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_state.layout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "Scene Capture",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_state.layout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  const wgpu::ColorTargetState target{.format = CopyFormat, .writeMask = wgpu::ColorWriteMask::All};
  const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs_main", .targetCount = 1, .targets = &target};
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Scene Capture",
      .layout = pipelineLayout,
      .vertex = {.module = module, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .fragment = &fragment,
  };
  g_state.pipeline = g_device.CreateRenderPipeline(&descriptor);
  g_state.samples = samples;
}

void ensure_copy(uint32_t width, uint32_t height) {
  // A 3D texture's sides are capped by the device (2048 on D3D12's baseline): scale the copy down
  // to fit, keeping the frame's aspect; the reflections it feeds need no more.
  wgpu::Limits limits{};
  g_device.GetLimits(&limits);
  const uint32_t cap = std::max<uint32_t>(limits.maxTextureDimension3D, 256);
  if (width > cap || height > cap) {
    const double s = double(cap) / double(std::max(width, height));
    width = std::max<uint32_t>(1, uint32_t(double(width) * s));
    height = std::max<uint32_t>(1, uint32_t(double(height) * s));
  }
  if (g_state.copy && g_state.width == width && g_state.height == height) {
    return;
  }
  g_state.width = width;
  g_state.height = height;
  const wgpu::TextureDescriptor descriptor{
      .label = "Scene Capture",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e3D,
      .size = {width, height, 1},
      .format = CopyFormat,
  };
  g_state.copy = g_device.CreateTexture(&descriptor);
  g_state.copyView = g_state.copy.CreateView();
  g_state.copyTarget = g_state.copy.CreateView();
}

void ensure_frame(uint32_t width, uint32_t height, wgpu::TextureFormat format) {
  if (g_state.frame && g_state.frame.GetWidth() == width && g_state.frame.GetHeight() == height &&
      g_state.frameFormat == format) {
    return;
  }
  g_state.frameFormat = format;
  const wgpu::TextureDescriptor descriptor{
      .label = "Scene Capture Frame Copy",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .size = {width, height, 1},
      .format = format,
  };
  g_state.frame = g_device.CreateTexture(&descriptor);
  g_state.frameView = g_state.frame.CreateView();
}

void encode(const EncoderTaskContext& ctx, const wgpu::CommandEncoder& cmd, const void* payload, size_t payloadSize,
            void*) {
  if (payloadSize != sizeof(Params)) {
    return;
  }
  Params params;
  std::memcpy(&params, payload, sizeof(params));
  const auto& source = webgpu::present_source();
  const auto& depth = webgpu::g_depthBuffer;
  const uint32_t samples = webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_graphicsConfig.msaaSamples : 1;
  const uint32_t width = source.size.width;
  const uint32_t height = source.size.height;
  if (width == 0 || height == 0 || !source.texture || !depth.view || !g_state.copy) {
    return;
  }
  ensure_pipeline(samples);
  ensure_frame(width, height, webgpu::g_graphicsConfig.surfaceConfiguration.format);
  ctx.queue.WriteBuffer(g_state.uniforms, 0, &params, sizeof(params));
  // The pass reads the frame and writes its own target, so the frame is copied out first.
  const wgpu::TexelCopyTextureInfo copySource{.texture = source.texture};
  const wgpu::TexelCopyTextureInfo copyTarget{.texture = g_state.frame};
  const wgpu::Extent3D copySize{width, height, 1};
  cmd.CopyTextureToTexture(&copySource, &copyTarget, &copySize);
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = g_state.uniforms, .size = sizeof(Params)},
      wgpu::BindGroupEntry{.binding = 1, .textureView = g_state.frameView},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depth.view},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "Scene Capture",
      .layout = g_state.layout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto group = g_device.CreateBindGroup(&groupDescriptor);
  const wgpu::RenderPassColorAttachment attachment{
      .view = g_state.copyTarget,
      .depthSlice = 0,
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 1e4},
  };
  const wgpu::RenderPassDescriptor passDescriptor{
      .label = "Scene Capture",
      .colorAttachmentCount = 1,
      .colorAttachments = &attachment,
      .timestampWrites = webgpu::gpu_prof::pass_writes("Scene capture"),
  };
  const auto pass = cmd.BeginRenderPass(&passDescriptor);
  pass.SetPipeline(g_state.pipeline);
  pass.SetBindGroup(0, group);
  pass.Draw(3);
  pass.End();
}
} // namespace

bool ensure_task() {
  if (g_state.task == InvalidEncoderTask) {
    g_state.task = register_encoder_task_type(EncoderTaskDescriptor{.label = "Scene Capture", .callback = encode});
    if (g_state.task == InvalidEncoderTask) {
      Log.warn("could not register the scene capture task");
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
  // Made here, on the FIFO processor, so the draws after the capture bind this frame's copy.
  ensure_copy(size.width, size.height);
  return record_encoder_task(g_state.task, &params, sizeof(params));
}

const wgpu::TextureView& view() {
  if (g_state.copyView) {
    return g_state.copyView;
  }
  if (!g_state.placeholder) {
    const wgpu::TextureDescriptor descriptor{
        .label = "Scene Capture Placeholder",
        .usage = wgpu::TextureUsage::TextureBinding,
        .dimension = wgpu::TextureDimension::e3D,
        .size = {1, 1, 1},
        .format = CopyFormat,
    };
    g_state.placeholder = g_device.CreateTexture(&descriptor).CreateView();
  }
  return g_state.placeholder;
}

void shutdown() {
  const auto task = g_state.task;
  g_state = {};
  if (task != InvalidEncoderTask) {
    unregister_encoder_task_type(task);
  }
}
} // namespace aurora::gfx::scenecap
