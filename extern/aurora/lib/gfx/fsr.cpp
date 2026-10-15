#include "fsr.hpp"

#include "../logging.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "recording.hpp"

#include <aurora/gfx.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <dawn/native/D3D12Backend.h>
#include <windows.h>
#include <wrl/client.h>

#include "ffx/api/include/dx12/ffx_api_dx12.h"
#include "ffx/api/include/ffx_api.h"
#include "ffx/api/include/ffx_api_loader.h"
#include "ffx/upscalers/include/ffx_upscale.h"
#endif

namespace aurora::gfx::fsr {
namespace {
Module Log("aurora::gfx::fsr");
using webgpu::g_device;

constexpr uint32_t Group = 8;

// The inputs, made from the frame at its end: the colour as presented, the device depth, and the
// motion from each pixel back to where it was in the last frame (pixels of the render size).
constexpr const char* PrepSource = R"(
struct P {
  view: mat3x4f,
  prevView: mat3x4f,
  frustum: vec4f,
  depth: vec4f,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var depthTex: DEPTH_TYPE;
@group(0) @binding(2) var finalTex: texture_2d<f32>;
@group(0) @binding(3) var opaqueTex: texture_2d<f32>;
@group(0) @binding(4) var outColor: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var outDepth: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var outMotion: texture_storage_2d<rg32float, write>;

fn linear_of(z: f32) -> f32 {
  let d = clamp((z - p.depth.z) / max(p.depth.w - p.depth.z, 1e-6), 0.0, 1.0);
  return p.depth.x * p.depth.y / (p.depth.y - d * (p.depth.y - p.depth.x));
}

fn view_ray(uv: vec2f) -> vec3f {
  return vec3f(mix(p.frustum.x, p.frustum.y, uv.x), mix(p.frustum.z, p.frustum.w, 1.0 - uv.y), -1.0);
}

fn project_prev(world: vec3f) -> vec2f {
  let pe = vec3f(p.prevView[0].w, p.prevView[1].w, p.prevView[2].w);
  let r = mat3x3f(p.prevView[0].xyz, p.prevView[1].xyz, p.prevView[2].xyz);
  let v = r * (world - pe);
  let zl = max(-v.z, 1e-4);
  let ray = v.xy / zl;
  let s = (ray - vec2f(p.frustum.x, p.frustum.z)) / vec2f(p.frustum.y - p.frustum.x, p.frustum.w - p.frustum.z);
  return vec2f(s.x, 1.0 - s.y);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(textureDimensions(outColor));
  let px = vec2i(gid.xy);
  if (any(px >= size)) {
    return;
  }
  let c = textureLoad(finalTex, px, 0);
  let o = textureLoad(opaqueTex, px, 0);
  textureStore(outColor, px, vec4f(c.rgb, 1.0));
  let raw = textureLoad(depthTex, px, 0);
  textureStore(outDepth, px, vec4f(raw, 0.0, 0.0, 0.0));
  let z = 1.0 - raw;
  // The gun and the HUD don't move with the world: no motion.
  let hud = max(max(abs(c.r - o.r), abs(c.g - o.g)), abs(c.b - o.b)) > 0.012;
  if (z < p.depth.z || hud) {
    textureStore(outMotion, px, vec4f(0.0));
    return;
  }
  var zl = p.depth.y;
  if (z < 0.99999) {
    zl = linear_of(z);
  }
  let fsize = vec2f(size);
  let uv = (vec2f(px) + 0.5) / fsize;
  let world = vec4f(view_ray(uv) * zl, 1.0) * p.view;
  let prev = project_prev(world);
  textureStore(outMotion, px, vec4f((prev - uv) * fsize, 0.0, 0.0));
}
)";

struct PrepUniform {
  float view[12];
  float prevView[12];
  float frustum[4];
  float depth[4];
};

struct State {
  EncoderTaskId task = InvalidEncoderTask;
  // The frame before the HUD (the capture task's copy).
  wgpu::Texture opaque;
  wgpu::TextureView opaqueView;
  uint32_t opaqueWidth = 0, opaqueHeight = 0;
  wgpu::TextureFormat opaqueFormat = wgpu::TextureFormat::Undefined;
  Params params{};
  bool captured = false;
  // The prep pass.
  wgpu::Buffer uniforms;
  wgpu::BindGroupLayout prepLayout;
  wgpu::ComputePipeline prep;
  uint32_t prepSamples = 0;
  bool loggedFirst = false;
};
State g_state;

std::array<Params, 8> g_recorded;
uint32_t g_nextSlot = 0;

float halton(uint32_t index, uint32_t base) {
  float f = 1.f, r = 0.f;
  while (index > 0) {
    f /= float(base);
    r += f * float(index % base);
    index /= base;
  }
  return r;
}

void capture(const EncoderTaskContext&, const wgpu::CommandEncoder& cmd, const void* payload, size_t payloadSize,
             void*) {
  uint32_t slot = 0;
  if (payloadSize != sizeof(slot)) {
    return;
  }
  std::memcpy(&slot, payload, sizeof(slot));
  const auto& source = webgpu::present_source();
  if (!source.texture || source.size.width == 0 || source.size.height == 0) {
    return;
  }
  if (!g_state.opaque || g_state.opaqueWidth != source.size.width || g_state.opaqueHeight != source.size.height ||
      g_state.opaqueFormat != source.format) {
    const wgpu::TextureDescriptor descriptor{
        .label = "FSR Frame Before HUD",
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
        .size = {source.size.width, source.size.height, 1},
        .format = source.format,
    };
    g_state.opaque = g_device.CreateTexture(&descriptor);
    g_state.opaqueView = g_state.opaque.CreateView();
    g_state.opaqueWidth = source.size.width;
    g_state.opaqueHeight = source.size.height;
    g_state.opaqueFormat = source.format;
  }
  const wgpu::TexelCopyTextureInfo from{.texture = source.texture};
  const wgpu::TexelCopyTextureInfo to{.texture = g_state.opaque};
  const wgpu::Extent3D extent{source.size.width, source.size.height, 1};
  cmd.CopyTextureToTexture(&from, &to, &extent);
  g_state.params = g_recorded[slot % g_recorded.size()];
  g_state.captured = true;
}

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;

template <typename F>
bool checked(const char* what, F&& fn) {
  g_device.PushErrorScope(wgpu::ErrorFilter::Validation);
  fn();
  bool ok = true;
  const auto future = g_device.PopErrorScope(
      wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::PopErrorScopeStatus, wgpu::ErrorType type, wgpu::StringView message) {
        if (type != wgpu::ErrorType::NoError) {
          Log.warn("{}: {}", what, std::string_view(message.data, message.length));
          ok = false;
        }
      });
  webgpu::g_instance.WaitAny(future, 5000000000);
  return ok;
}

struct Shared {
  ComPtr<ID3D12Resource> resource;
  wgpu::SharedTextureMemory memory;
  wgpu::Texture texture;
  wgpu::TextureView view;
};

struct D3D {
  bool tried = false;
  bool ok = false;
  HMODULE module = nullptr;
  ffxFunctions ffx{};
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocators[3];
  uint64_t allocatorFence[3] = {};
  uint32_t allocator = 0;
  ComPtr<ID3D12GraphicsCommandList> list;
  ComPtr<ID3D12Fence> fence;
  uint64_t fenceValue = 0;
  HANDLE event = nullptr;
  ffxContext context = nullptr;
  uint32_t renderWidth = 0, renderHeight = 0, displayWidth = 0, displayHeight = 0;
  Shared color, depth, motion, output;
  webgpu::TextureWithSampler outputTex;
  bool failed = false; // a dispatch failed: no more tries this run
};
D3D g_d3d;

void wait_fence(uint64_t value) {
  if (value == 0 || g_d3d.fence->GetCompletedValue() >= value) {
    return;
  }
  g_d3d.fence->SetEventOnCompletion(value, g_d3d.event);
  WaitForSingleObject(g_d3d.event, 5000);
}

void message(uint32_t type, const wchar_t* text) {
  if (text == nullptr) {
    return;
  }
  std::string s;
  for (const wchar_t* c = text; *c != 0 && s.size() < 512; ++c) {
    s.push_back(*c < 128 ? char(*c) : '?');
  }
  if (type == FFX_API_MESSAGE_TYPE_ERROR) {
    Log.warn("FSR: {}", s);
  } else {
    Log.info("FSR: {}", s);
  }
}

bool init() {
  if (g_d3d.tried) {
    return g_d3d.ok;
  }
  if (!g_device) {
    return false; // asked before the device exists: ask again later
  }
  g_d3d.tried = true;
  if (webgpu::g_backendType != wgpu::BackendType::D3D12) {
    Log.info("FSR needs the D3D12 backend");
    return false;
  }
  if (!g_device.HasFeature(wgpu::FeatureName::SharedTextureMemoryD3D12Resource)) {
    Log.warn("FSR: Dawn device lacks SharedTextureMemoryD3D12Resource");
    return false;
  }
  g_d3d.module = LoadLibraryW(L"amd_fidelityfx_loader_dx12.dll");
  if (g_d3d.module == nullptr) {
    Log.warn("FSR: amd_fidelityfx_loader_dx12.dll not found beside the executable");
    return false;
  }
  ffxLoadFunctions(&g_d3d.ffx, g_d3d.module);
  if (!g_d3d.ffx.CreateContext || !g_d3d.ffx.Dispatch || !g_d3d.ffx.DestroyContext || !g_d3d.ffx.Query) {
    Log.warn("FSR: the loader DLL lacks its entry points");
    return false;
  }
  g_d3d.device = dawn::native::d3d12::GetD3D12Device(g_device.Get());
  g_d3d.queue = dawn::native::d3d12::GetD3D12CommandQueue(g_device.Get());
  if (!g_d3d.device || !g_d3d.queue) {
    Log.warn("FSR: no D3D12 device or queue from Dawn");
    return false;
  }
  for (auto& a : g_d3d.allocators) {
    if (FAILED(g_d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a)))) {
      Log.warn("FSR: command allocator");
      return false;
    }
  }
  if (FAILED(g_d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_d3d.allocators[0].Get(), nullptr,
                                             IID_PPV_ARGS(&g_d3d.list))) ||
      FAILED(g_d3d.list->Close()) ||
      FAILED(g_d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_d3d.fence)))) {
    Log.warn("FSR: command list or fence");
    return false;
  }
  g_d3d.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  g_d3d.ok = true;
  Log.info("FSR loader ready");
  return true;
}

void drop_shared(Shared& s) {
  if (s.memory && s.texture) {
    checked("FSR shared texture end", [&] {
      wgpu::SharedTextureMemoryEndAccessState state{};
      s.memory.EndAccess(s.texture, &state);
    });
  }
  s = {};
}

bool make_shared(Shared& s, uint32_t width, uint32_t height, DXGI_FORMAT format, const char* label) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = width;
  desc.Height = height;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = format;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  // As the ray tracing's: simultaneous access, so Dawn's command lists and FSR's share them on
  // Dawn's queue without handing layouts over.
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
  if (FAILED(g_d3d.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                   nullptr, IID_PPV_ARGS(&s.resource)))) {
    Log.warn("FSR: {} texture", label);
    return false;
  }
  return checked(label, [&] {
    dawn::native::d3d12::SharedTextureMemoryD3D12ResourceDescriptor d3dDescriptor;
    d3dDescriptor.resource = s.resource;
    wgpu::SharedTextureMemoryDescriptor descriptor{};
    descriptor.nextInChain = &d3dDescriptor;
    descriptor.label = label;
    s.memory = g_device.ImportSharedTextureMemory(&descriptor);
    s.texture = s.memory.CreateTexture();
    wgpu::SharedTextureMemoryBeginAccessDescriptor begin{};
    begin.initialized = true;
    s.memory.BeginAccess(s.texture, &begin);
    s.view = s.texture.CreateView();
  });
}

void destroy_context() {
  if (g_d3d.context != nullptr) {
    wait_fence(g_d3d.fenceValue);
    g_d3d.ffx.DestroyContext(&g_d3d.context, nullptr);
    g_d3d.context = nullptr;
  }
}

bool ensure_size(uint32_t rw, uint32_t rh, uint32_t dw, uint32_t dh) {
  if (g_d3d.context != nullptr && g_d3d.renderWidth == rw && g_d3d.renderHeight == rh && g_d3d.displayWidth == dw &&
      g_d3d.displayHeight == dh) {
    return true;
  }
  wait_fence(g_d3d.fenceValue);
  destroy_context();
  drop_shared(g_d3d.color);
  drop_shared(g_d3d.depth);
  drop_shared(g_d3d.motion);
  drop_shared(g_d3d.output);
  g_d3d.outputTex = {};
  g_d3d.renderWidth = g_d3d.renderHeight = g_d3d.displayWidth = g_d3d.displayHeight = 0;
  if (!make_shared(g_d3d.color, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, "FSR Colour") ||
      !make_shared(g_d3d.depth, rw, rh, DXGI_FORMAT_R32_FLOAT, "FSR Depth") ||
      !make_shared(g_d3d.motion, rw, rh, DXGI_FORMAT_R32G32_FLOAT, "FSR Motion") ||
      !make_shared(g_d3d.output, dw, dh, DXGI_FORMAT_R16G16B16A16_FLOAT, "FSR Output")) {
    return false;
  }
  ffxCreateBackendDX12Desc backend{};
  backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
  backend.device = g_d3d.device.Get();
  ffxCreateContextDescUpscaleVersion version{};
  version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
  version.header.pNext = &backend.header;
  version.version = FFX_UPSCALER_VERSION;
  ffxCreateContextDescUpscale desc{};
  desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
  desc.header.pNext = &version.header;
  desc.flags = FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE |
               FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE;
  desc.maxRenderSize = {rw, rh};
  desc.maxUpscaleSize = {dw, dh};
  desc.fpMessage = message;
  const ffxReturnCode_t rc = g_d3d.ffx.CreateContext(&g_d3d.context, &desc.header, nullptr);
  if (rc != FFX_API_RETURN_OK || g_d3d.context == nullptr) {
    Log.warn("FSR: context creation failed ({})", rc);
    g_d3d.context = nullptr;
    return false;
  }
  ffxQueryGetProviderVersion provider{};
  provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
  if (g_d3d.ffx.Query(&g_d3d.context, &provider.header) == FFX_API_RETURN_OK && provider.versionName != nullptr) {
    Log.info("FSR {} upscaling {}x{} -> {}x{}", provider.versionName, rw, rh, dw, dh);
  }
  g_d3d.renderWidth = rw;
  g_d3d.renderHeight = rh;
  g_d3d.displayWidth = dw;
  g_d3d.displayHeight = dh;
  g_d3d.outputTex.texture = g_d3d.output.texture;
  g_d3d.outputTex.view = g_d3d.output.view;
  g_d3d.outputTex.size = {dw, dh, 1};
  g_d3d.outputTex.format = wgpu::TextureFormat::RGBA16Float;
  g_d3d.outputTex.sampler = webgpu::present_source().sampler;
  return true;
}

void ensure_prep(uint32_t samples) {
  if (!g_state.uniforms) {
    const wgpu::BufferDescriptor descriptor{
        .label = "FSR Prep Uniforms",
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(PrepUniform),
    };
    g_state.uniforms = g_device.CreateBuffer(&descriptor);
  }
  if (g_state.prep && g_state.prepSamples == samples) {
    return;
  }
  const auto cs = wgpu::ShaderStage::Compute;
  const auto storage = [&](uint32_t binding, wgpu::TextureFormat format) {
    return wgpu::BindGroupLayoutEntry{
        .binding = binding,
        .visibility = cs,
        .storageTexture = {.access = wgpu::StorageTextureAccess::WriteOnly,
                           .format = format,
                           .viewDimension = wgpu::TextureViewDimension::e2D},
    };
  };
  const auto sampled = [&](uint32_t binding) {
    return wgpu::BindGroupLayoutEntry{
        .binding = binding,
        .visibility = cs,
        .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                    .viewDimension = wgpu::TextureViewDimension::e2D},
    };
  };
  const std::array entries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = cs,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(PrepUniform)},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = cs,
          .texture = {.sampleType = wgpu::TextureSampleType::Depth,
                      .viewDimension = wgpu::TextureViewDimension::e2D,
                      .multisampled = samples > 1},
      },
      sampled(2),
      sampled(3),
      storage(4, wgpu::TextureFormat::RGBA16Float),
      storage(5, wgpu::TextureFormat::R32Float),
      storage(6, wgpu::TextureFormat::RG32Float),
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "FSR Prep Layout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_state.prepLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "FSR Prep",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_state.prepLayout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  std::string code = PrepSource;
  const std::string token = "DEPTH_TYPE";
  code.replace(code.find(token), token.size(), samples > 1 ? "texture_depth_multisampled_2d" : "texture_depth_2d");
  wgpu::ShaderSourceWGSL source{};
  source.code = code.c_str();
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "FSR Prep"};
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);
  const wgpu::ComputePipelineDescriptor descriptor{
      .label = "FSR Prep",
      .layout = pipelineLayout,
      .compute = {.module = module, .entryPoint = "cs_main"},
  };
  g_state.prep = g_device.CreateComputePipeline(&descriptor);
  g_state.prepSamples = samples;
}

FfxApiResource resource_of(const Shared& s) {
  return ffxApiGetResourceDX12(s.resource.Get(), FFX_API_RESOURCE_STATE_COMMON);
}
#endif
} // namespace

float ratio(uint32_t mode) {
  switch (mode) {
  case Quality:
    return 1.5f;
  case Balanced:
    return 1.7f;
  case Performance:
    return 2.f;
  default:
    return 1.f;
  }
}

bool available() {
#if defined(_WIN32)
  return init();
#else
  return false;
#endif
}

void jitter(uint64_t index, uint32_t mode, float& x, float& y) {
  // FSR's own sequence: Halton (2, 3), 8 r^2 phases for a ratio r.
  const float r = ratio(mode);
  const uint32_t phases = std::max(8u, uint32_t(std::ceil(8.f * r * r)));
  const uint32_t i = uint32_t(index % phases) + 1;
  x = halton(i, 2) - 0.5f;
  y = halton(i, 3) - 0.5f;
}

bool ensure_task() {
  if (g_state.task == InvalidEncoderTask) {
    g_state.task = register_encoder_task_type(EncoderTaskDescriptor{.label = "FSR Capture", .callback = capture});
    if (g_state.task == InvalidEncoderTask) {
      Log.warn("could not register the FSR capture task");
      return false;
    }
  }
  return true;
}

bool record(const Params& params) {
  if (g_state.task == InvalidEncoderTask || params.mode == Off) {
    return false;
  }
  const uint32_t slot = g_nextSlot++ % g_recorded.size();
  g_recorded[slot] = params;
  return record_encoder_task(g_state.task, &slot, sizeof(slot));
}

const webgpu::TextureWithSampler* upscale(wgpu::CommandEncoder& encoder) {
  if (!g_state.captured) {
    return nullptr;
  }
  g_state.captured = false;
#if defined(_WIN32)
  const Params params = g_state.params;
  if (params.mode == Off || g_d3d.failed || !init()) {
    return nullptr;
  }
  const auto& source = webgpu::present_source();
  const auto& depthBuffer = webgpu::g_depthBuffer;
  if (!source.texture || !depthBuffer.view || !g_state.opaqueView) {
    return nullptr;
  }
  const uint32_t rw = source.size.width;
  const uint32_t rh = source.size.height;
  const float r = ratio(params.mode);
  const uint32_t dw = std::max<uint32_t>(1, uint32_t(std::lround(float(rw) * r)));
  const uint32_t dh = std::max<uint32_t>(1, uint32_t(std::lround(float(rh) * r)));
  if (g_state.opaqueWidth != rw || g_state.opaqueHeight != rh) {
    return nullptr; // resized between the capture and now
  }
  if (!ensure_size(rw, rh, dw, dh)) {
    g_d3d.failed = true;
    Log.warn("FSR off for this run");
    return nullptr;
  }
  const uint32_t samples = webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_graphicsConfig.msaaSamples : 1;
  ensure_prep(samples);

  // The inputs.
  PrepUniform u{};
  std::memcpy(u.view, params.view, sizeof(u.view));
  std::memcpy(u.prevView, params.prevView, sizeof(u.prevView));
  std::memcpy(u.frustum, params.frustum, sizeof(u.frustum));
  std::memcpy(u.depth, params.depth, sizeof(u.depth));
  webgpu::g_queue.WriteBuffer(g_state.uniforms, 0, &u, sizeof(u));
  {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .buffer = g_state.uniforms, .size = sizeof(PrepUniform)},
        wgpu::BindGroupEntry{.binding = 1, .textureView = depthBuffer.view},
        wgpu::BindGroupEntry{.binding = 2, .textureView = source.view},
        wgpu::BindGroupEntry{.binding = 3, .textureView = g_state.opaqueView},
        wgpu::BindGroupEntry{.binding = 4, .textureView = g_d3d.color.view},
        wgpu::BindGroupEntry{.binding = 5, .textureView = g_d3d.depth.view},
        wgpu::BindGroupEntry{.binding = 6, .textureView = g_d3d.motion.view},
    };
    const wgpu::BindGroupDescriptor groupDescriptor{
        .label = "FSR Prep",
        .layout = g_state.prepLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    const auto group = g_device.CreateBindGroup(&groupDescriptor);
    const wgpu::ComputePassDescriptor passDescriptor{.label = "FSR Prep"};
    const auto pass = encoder.BeginComputePass(&passDescriptor);
    pass.SetPipeline(g_state.prep);
    pass.SetBindGroup(0, group);
    pass.DispatchWorkgroups((rw + Group - 1) / Group, (rh + Group - 1) / Group, 1);
    pass.End();
  }
  // The frame so far goes; FSR follows it on the same queue.
  {
    const wgpu::CommandBufferDescriptor bufferDescriptor{.label = "Frame before FSR"};
    const auto buffer = encoder.Finish(&bufferDescriptor);
    webgpu::g_queue.Submit(1, &buffer);
    const wgpu::CommandEncoderDescriptor encoderDescriptor{.label = "Frame after FSR"};
    encoder = g_device.CreateCommandEncoder(&encoderDescriptor);
  }
  const uint32_t a = g_d3d.allocator;
  wait_fence(g_d3d.allocatorFence[a]);
  if (FAILED(g_d3d.allocators[a]->Reset()) || FAILED(g_d3d.list->Reset(g_d3d.allocators[a].Get(), nullptr))) {
    return nullptr;
  }
  ffxDispatchDescUpscale d{};
  d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
  d.commandList = g_d3d.list.Get();
  d.color = resource_of(g_d3d.color);
  d.depth = resource_of(g_d3d.depth);
  d.motionVectors = resource_of(g_d3d.motion);
  d.output = ffxApiGetResourceDX12(g_d3d.output.resource.Get(), FFX_API_RESOURCE_STATE_COMMON,
                                   FFX_API_RESOURCE_USAGE_UAV);
  d.jitterOffset = {params.jitter[0], params.jitter[1]};
  d.motionVectorScale = {1.f, 1.f};
  d.renderSize = {rw, rh};
  d.upscaleSize = {dw, dh};
  d.enableSharpening = true;
  d.sharpness = 0.3f;
  d.frameTimeDelta = params.frameMs > 0.f ? params.frameMs : 16.6f;
  d.preExposure = 1.f;
  d.reset = params.reset != 0;
  d.cameraNear = params.depth[0];
  d.cameraFar = params.depth[1];
  d.cameraFovAngleVertical = params.vfov > 0.f ? params.vfov : 1.2f;
  d.viewSpaceToMetersFactor = 1.f;
  d.flags = FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
  const ffxReturnCode_t rc = g_d3d.ffx.Dispatch(&g_d3d.context, &d.header);
  if (FAILED(g_d3d.list->Close()) || rc != FFX_API_RETURN_OK) {
    Log.warn("FSR: dispatch failed ({}); FSR off for this run", rc);
    g_d3d.failed = true;
    return nullptr;
  }
  ID3D12CommandList* lists[] = {g_d3d.list.Get()};
  g_d3d.queue->ExecuteCommandLists(1, lists);
  g_d3d.queue->Signal(g_d3d.fence.Get(), ++g_d3d.fenceValue);
  g_d3d.allocatorFence[a] = g_d3d.fenceValue;
  g_d3d.allocator = (a + 1) % 3;
  if (!g_state.loggedFirst) {
    g_state.loggedFirst = true;
    Log.info("FSR: first frame upscaled");
  }
  return &g_d3d.outputTex;
#else
  (void)encoder;
  return nullptr;
#endif
}

void shutdown() {
#if defined(_WIN32)
  if (g_d3d.ok) {
    wait_fence(g_d3d.fenceValue);
    destroy_context();
    drop_shared(g_d3d.color);
    drop_shared(g_d3d.depth);
    drop_shared(g_d3d.motion);
    drop_shared(g_d3d.output);
  }
  if (g_d3d.event != nullptr) {
    CloseHandle(g_d3d.event);
  }
  g_d3d = {};
#endif
  const auto task = g_state.task;
  g_state = {};
  if (task != InvalidEncoderTask) {
    unregister_encoder_task_type(task);
  }
}
} // namespace aurora::gfx::fsr
