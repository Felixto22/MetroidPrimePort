#include "rt.hpp"

#include "../logging.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "recording.hpp"
#include "shadow.hpp"

#include <aurora/gfx.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <dawn/native/D3D12Backend.h>
#include <windows.h>
#endif

// Ray-traced sun shadows (see rt.hpp). The Dawn side (the depth copy and the shadow map layer)
// builds everywhere; the D3D12 side only on Windows, where available() can be true.
namespace aurora::gfx::rt {
namespace {
Module Log("aurora::gfx::rt");
using webgpu::g_device;

constexpr uint32_t Group = 8;
// The shadow map layer's depth unit: a texel packs round(depth * 64) (16 bits) above its
// visibility (8 bits), as a 24-bit integer over 2^24, which a Depth32Float holds exactly.
constexpr float DepthUnits = 64.f;

// What the trace needs from the frame (any thread).
struct SharedState {
  std::mutex mutex;
  std::unordered_map<uint64_t, std::vector<float>> pending; // meshes waiting for the render thread
  std::unordered_set<uint64_t> known;                       // pending or built
  std::vector<uint64_t> scene;
  std::vector<uint64_t> drop; // meshes the game forgot
};
SharedState& shared() {
  static SharedState state;
  return state;
}

// FIFO processor: the frame traced last, for the next frame's receivers.
struct FifoState {
  bool traced = false;
  float worldToView[3][4]{};
  float frustum[4]{};
  uint32_t width = 0;
  uint32_t height = 0;
};
FifoState g_fifo;

std::atomic<bool> g_ready{false};
std::atomic<bool> g_failed{false};
EncoderTaskId g_task = InvalidEncoderTask;

// Render thread.
struct RenderState {
  wgpu::Buffer uniforms;
  wgpu::BindGroupLayout copyLayout;
  wgpu::ComputePipeline copy;
  uint32_t copySamples = 0;
  wgpu::BindGroupLayout injectLayout;
  wgpu::RenderPipeline inject;
  // The textures shared with D3D12: the world's linear depth (Dawn writes, D3D12 reads) and the
  // traced visibility (D3D12 writes, Dawn reads).
  wgpu::SharedTextureMemory depthMemory;
  wgpu::Texture depthTexture;
  wgpu::TextureView depthView;
  wgpu::SharedTextureMemory maskMemory;
  wgpu::Texture maskTexture;
  wgpu::TextureView maskView;
  uint32_t width = 0;
  uint32_t height = 0;
  bool depthCopied = false; // this frame's depth is in depthTexture, for the trace after submit
  bool traced = false;      // the mask holds the last frame's trace, for the next begin
};
RenderState g_render;

constexpr const char* CopySource = R"(
struct P {
  depth: vec4f,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var depthTex: DEPTH_TYPE;
@group(0) @binding(2) var outTex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let osize = textureDimensions(outTex);
  if (gid.x >= osize.x || gid.y >= osize.y) {
    return;
  }
  let size = vec2i(textureDimensions(depthTex));
  let at = min(vec2i((vec2f(gid.xy) + 0.5) * vec2f(size) / vec2f(osize)), size - vec2i(1));
  let z = 1.0 - textureLoad(depthTex, at, 0);
  var zl = 0.0;
  if (z >= p.depth.z && z < 0.99999) {
    let d = clamp((z - p.depth.z) / max(p.depth.w - p.depth.z, 1e-6), 0.0, 1.0);
    zl = p.depth.x * p.depth.y / (p.depth.y - d * (p.depth.y - p.depth.x));
  }
  textureStore(outTex, gid.xy, vec4f(zl, 0.0, 0.0, 0.0));
}
)";

// The traced visibility, blurred over texels at about the same depth (the rays are few), packed
// with the depth into the shadow map's extra layer.
constexpr const char* InjectSource = R"(
@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var maskTex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(corners[i], 0.0, 1.0);
}

struct Out {
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> Out {
  var out: Out;
  out.depth = 0.0;
  let size = vec2i(textureDimensions(depthTex));
  let p = vec2i(floor(pos.xy));
  if (any(p >= size)) {
    return out;
  }
  let z = textureLoad(depthTex, p, 0).r;
  if (z <= 0.0) {
    return out;
  }
  var sum = 0.0;
  var weight = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let q = clamp(p + vec2i(x, y), vec2i(0), size - vec2i(1));
      let zq = textureLoad(depthTex, q, 0).r;
      if (zq > 0.0 && abs(zq - z) < 0.03 * z + 0.05) {
        let w = f32((3 - abs(x)) * (3 - abs(y)));
        sum += textureLoad(maskTex, q, 0).r * w;
        weight += w;
      }
    }
  }
  let vis = clamp(sum / max(weight, 1e-4), 0.0, 1.0);
  let dq = clamp(round(z * DEPTH_UNITS), 1.0, 65535.0);
  out.depth = (dq * 256.0 + round(vis * 255.0)) / 16777216.0;
  return out;
}
)";

std::string replace_token(std::string code, const std::string& token, const std::string& with) {
  for (size_t at = code.find(token); at != std::string::npos; at = code.find(token, at + with.size())) {
    code.replace(at, token.size(), with);
  }
  return code;
}

wgpu::ShaderModule make_module(const char* label, const std::string& code) {
  wgpu::ShaderSourceWGSL source{};
  source.code = code.c_str();
  const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &source, .label = label};
  return g_device.CreateShaderModule(&descriptor);
}

void ensure_copy(uint32_t samples) {
  if (!g_render.uniforms) {
    const wgpu::BufferDescriptor descriptor{
        .label = "RT Depth Copy Uniforms",
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = 16,
    };
    g_render.uniforms = g_device.CreateBuffer(&descriptor);
  }
  if (g_render.copy && g_render.copySamples == samples) {
    return;
  }
  const std::array entries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = 16},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Compute,
          .texture = {.sampleType = wgpu::TextureSampleType::Depth,
                      .viewDimension = wgpu::TextureViewDimension::e2D,
                      .multisampled = samples > 1},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Compute,
          .storageTexture = {.access = wgpu::StorageTextureAccess::WriteOnly,
                             .format = wgpu::TextureFormat::R32Float,
                             .viewDimension = wgpu::TextureViewDimension::e2D},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "RT Depth Copy Layout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_render.copyLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "RT Depth Copy",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_render.copyLayout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  const auto module = make_module(
      "RT Depth Copy",
      replace_token(CopySource, "DEPTH_TYPE", samples > 1 ? "texture_depth_multisampled_2d" : "texture_depth_2d"));
  const wgpu::ComputePipelineDescriptor descriptor{
      .label = "RT Depth Copy",
      .layout = pipelineLayout,
      .compute = {.module = module, .entryPoint = "cs_main"},
  };
  g_render.copy = g_device.CreateComputePipeline(&descriptor);
  g_render.copySamples = samples;
}

void ensure_inject() {
  if (g_render.inject) {
    return;
  }
  const std::array entries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "RT Shadow Inject Layout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_render.injectLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "RT Shadow Inject",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_render.injectLayout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  char units[32];
  std::snprintf(units, sizeof(units), "%.1f", DepthUnits);
  const auto module = make_module("RT Shadow Inject", replace_token(InjectSource, "DEPTH_UNITS", units));
  const wgpu::DepthStencilState depthState{
      .format = shadow::MapFormat,
      .depthWriteEnabled = wgpu::OptionalBool::True,
      .depthCompare = wgpu::CompareFunction::Always,
  };
  const wgpu::FragmentState fragment{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = 0,
      .targets = nullptr,
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "RT Shadow Inject",
      .layout = pipelineLayout,
      .vertex = {.module = module, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .depthStencil = &depthState,
      .multisample = {.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  g_render.inject = g_device.CreateRenderPipeline(&descriptor);
}

// Runs fn inside a validation error scope; false (with the message logged) when Dawn objected.
template <typename F>
bool checked(const char* what, F&& fn) {
  g_device.PushErrorScope(wgpu::ErrorFilter::Validation);
  fn();
  bool failed = false;
  std::string error;
  const auto future = g_device.PopErrorScope(
      wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::PopErrorScopeStatus, wgpu::ErrorType type, wgpu::StringView message) {
        if (type != wgpu::ErrorType::NoError) {
          failed = true;
          error = std::string{std::string_view{message}};
        }
      });
  webgpu::g_instance.WaitAny(future, 5000000000);
  if (failed) {
    Log.warn("{}: {}", what, error);
  }
  return !failed;
}

void disable(const char* why) {
  if (!g_failed.exchange(true)) {
    Log.warn("ray-traced shadows off: {}", why);
  }
  g_ready.store(false);
}

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;

// The few DXC interfaces used (dxcapi.h's), so the build doesn't need the SDK's copy.
struct DxcBufferRt {
  const void* Ptr;
  SIZE_T Size;
  UINT Encoding;
};
struct __declspec(novtable) IDxcBlobRt : public IUnknown {
  virtual LPVOID STDMETHODCALLTYPE GetBufferPointer() = 0;
  virtual SIZE_T STDMETHODCALLTYPE GetBufferSize() = 0;
};
struct __declspec(novtable) IDxcBlobEncodingRt : public IDxcBlobRt {
  virtual HRESULT STDMETHODCALLTYPE GetEncoding(BOOL* known, UINT32* codePage) = 0;
};
struct __declspec(novtable) IDxcOperationResultRt : public IUnknown {
  virtual HRESULT STDMETHODCALLTYPE GetStatus(HRESULT* status) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetResult(IDxcBlobRt** result) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetErrorBuffer(IDxcBlobEncodingRt** errors) = 0;
};
struct __declspec(novtable) IDxcCompiler3Rt : public IUnknown {
  virtual HRESULT STDMETHODCALLTYPE Compile(const DxcBufferRt* source, LPCWSTR* arguments, UINT32 argCount,
                                            IUnknown* includeHandler, REFIID riid, LPVOID* result) = 0;
  virtual HRESULT STDMETHODCALLTYPE Disassemble(const DxcBufferRt* object, REFIID riid, LPVOID* result) = 0;
};
constexpr GUID kClsidDxcCompiler = {0x73e22d93, 0xe6ce, 0x47f3, {0xb5, 0xbf, 0xf0, 0x66, 0x4f, 0x39, 0xc1, 0xb0}};
constexpr GUID kIidDxcCompiler3 = {0x228b4687, 0x5a6a, 0x4730, {0x90, 0x0c, 0x97, 0x02, 0xb2, 0x20, 0x3f, 0x54}};
constexpr GUID kIidDxcOperationResult = {0xcedb484a, 0xd4e9, 0x445a, {0xb9, 0x91, 0xca, 0x21, 0xca, 0x15, 0x7d, 0xc2}};
using DxcCreateInstanceProc = HRESULT(__stdcall*)(REFCLSID, REFIID, LPVOID*);

// The trace: per texel of the depth copy, its world point (and a normal from its neighbours'
// depths), nudged off the surface, casts four rays at the sun's disc; the share that escapes is
// its visibility. Inline ray queries (DXR 1.1) from a compute shader.
constexpr const char* TraceSource = R"hlsl(
#define RS "RootConstants(num32BitConstants=28, b0), SRV(t0), DescriptorTable(SRV(t1), UAV(u0))"
cbuffer C : register(b0) {
  float4 v2w0;
  float4 v2w1;
  float4 v2w2;
  float4 frustum; // left, right, bottom, top at a view depth of 1
  float4 sun;     // world direction to the sun, tan of the disc's radius
  float4 misc;    // offset off the surface, its growth with depth, ray length, 0
  float4 extra;
};
RaytracingAccelerationStructure scene : register(t0);
Texture2D<float> depthTex : register(t1);
RWTexture2D<float4> outTex : register(u0);

float3 view_pos(float2 uv, float zl) {
  float2 ray = float2(lerp(frustum.x, frustum.y, uv.x), lerp(frustum.z, frustum.w, 1.0 - uv.y));
  return float3(ray * zl, -zl);
}

float ign(float2 px) {
  return frac(52.9829189 * frac(dot(px, float2(0.06711056, 0.00583715))));
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint w, h;
  outTex.GetDimensions(w, h);
  if (id.x >= w || id.y >= h) {
    return;
  }
  int2 size = int2(w, h);
  int2 at = int2(id.xy);
  float zl = depthTex[at];
  if (zl <= 0.0) {
    outTex[id.xy] = float4(1.0, 0.0, 0.0, 1.0);
    return;
  }
  float2 texel = 1.0 / float2(size);
  float2 uv = (float2(at) + 0.5) * texel;
  float3 pos = view_pos(uv, zl);
  float zr = depthTex[min(at + int2(1, 0), size - 1)];
  float zlf = depthTex[max(at - int2(1, 0), int2(0, 0))];
  float zd = depthTex[min(at + int2(0, 1), size - 1)];
  float zu = depthTex[max(at - int2(0, 1), int2(0, 0))];
  float3 dx = float3(0.0, 0.0, 0.0);
  if (zr > 0.0 && (zlf <= 0.0 || abs(zr - zl) < abs(zlf - zl))) {
    dx = view_pos(uv + float2(texel.x, 0.0), zr) - pos;
  } else if (zlf > 0.0) {
    dx = pos - view_pos(uv - float2(texel.x, 0.0), zlf);
  }
  float3 dy = float3(0.0, 0.0, 0.0);
  if (zd > 0.0 && (zu <= 0.0 || abs(zd - zl) < abs(zu - zl))) {
    dy = view_pos(uv + float2(0.0, texel.y), zd) - pos;
  } else if (zu > 0.0) {
    dy = pos - view_pos(uv - float2(0.0, texel.y), zu);
  }
  float3 n = float3(0.0, 0.0, 1.0);
  if (dot(dx, dx) > 0.0 && dot(dy, dy) > 0.0) {
    n = normalize(cross(dy, dx));
  }
  if (dot(n, -pos) < 0.0) {
    n = -n;
  }
  float4 p4 = float4(pos, 1.0);
  float3 wp = float3(dot(v2w0, p4), dot(v2w1, p4), dot(v2w2, p4));
  float3 wn = normalize(float3(dot(v2w0.xyz, n), dot(v2w1.xyz, n), dot(v2w2.xyz, n)));
  float3 l = normalize(sun.xyz);
  // Off the surface along its normal, and a little toward the sun.
  float bias = misc.x + misc.y * zl;
  float3 origin = wp + wn * bias + l * (bias * 0.5);
  float3 up = abs(l.z) > 0.99 ? float3(1.0, 0.0, 0.0) : float3(0.0, 0.0, 1.0);
  float3 t = normalize(cross(up, l));
  float3 b = cross(l, t);
  float spin = ign(float2(id.xy)) * 6.2831853;
  const uint rays = 4;
  float lit = 0.0;
  [unroll]
  for (uint r = 0; r < rays; ++r) {
    float a = spin + float(r) * 1.5707963;
    float rad = sqrt((float(r) + 0.5) / float(rays)) * sun.w;
    float3 d = normalize(l + (t * cos(a) + b * sin(a)) * rad);
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = d;
    ray.TMin = 0.0;
    ray.TMax = misc.z;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
      lit += 1.0;
    }
  }
  outTex[id.xy] = float4(lit / float(rays), 0.0, 0.0, 1.0);
}
)hlsl";

constexpr uint32_t Slots = 3;
constexpr uint32_t MaxInstances = 64;

struct Mesh {
  ComPtr<ID3D12Resource> vertices;
  ComPtr<ID3D12Resource> blas;
  uint32_t triangles = 0;
  uint64_t lastUsed = 0;
};

struct D3D {
  ComPtr<ID3D12Device5> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12RootSignature> rootSignature;
  ComPtr<ID3D12PipelineState> pipeline;
  ComPtr<ID3D12DescriptorHeap> heap;
  UINT descriptorSize = 0;
  ComPtr<ID3D12Fence> fence;
  uint64_t fenceValue = 0;
  HANDLE event = nullptr;
  struct Slot {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12Resource> instances; // upload heap, MaxInstances descs
    uint64_t fence = 0;
  };
  std::array<Slot, Slots> slots;
  uint32_t slot = 0;
  ComPtr<ID3D12GraphicsCommandList4> list;
  ComPtr<ID3D12Resource> tlas;
  uint64_t tlasSize = 0;
  ComPtr<ID3D12Resource> scratch;
  uint64_t scratchSize = 0;
  ComPtr<ID3D12Resource> depth; // shared with Dawn (g_render.depthTexture)
  ComPtr<ID3D12Resource> mask;  // shared with Dawn (g_render.maskTexture)
  std::unordered_map<uint64_t, Mesh> meshes;
  // Resources the GPU may still read, freed once the fence passes their value.
  std::vector<std::pair<uint64_t, ComPtr<ID3D12Resource>>> garbage;
  uint64_t traces = 0;
};
D3D g_d3d;

std::string hresult(HRESULT hr) {
  char text[16];
  std::snprintf(text, sizeof(text), "0x%08lX", static_cast<unsigned long>(hr));
  return text;
}

ComPtr<ID3D12Resource> make_buffer(uint64_t size, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_FLAGS flags,
                                   D3D12_RESOURCE_STATES state) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = heapType;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = std::max<uint64_t>(size, 256);
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = flags;
  ComPtr<ID3D12Resource> buffer;
  const HRESULT hr =
      g_d3d.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&buffer));
  if (FAILED(hr)) {
    Log.warn("buffer of {} bytes: {}", size, hresult(hr));
    return {};
  }
  return buffer;
}

void wait_fence(uint64_t value) {
  if (value == 0 || g_d3d.fence->GetCompletedValue() >= value) {
    return;
  }
  g_d3d.fence->SetEventOnCompletion(value, g_d3d.event);
  WaitForSingleObject(g_d3d.event, 5000);
}

void collect_garbage() {
  const uint64_t done = g_d3d.fence->GetCompletedValue();
  std::erase_if(g_d3d.garbage, [done](const auto& entry) { return entry.first <= done; });
}

void retire(ComPtr<ID3D12Resource>& resource) {
  if (resource) {
    g_d3d.garbage.emplace_back(g_d3d.fenceValue + 1, std::move(resource));
    resource.Reset();
  }
}

bool compile_trace(ComPtr<ID3D12RootSignature>& rootSignature, ComPtr<ID3D12PipelineState>& pipeline) {
  HMODULE dxc = LoadLibraryW(L"dxcompiler.dll");
  if (dxc == nullptr) {
    Log.warn("dxcompiler.dll not found");
    return false;
  }
  const auto create = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(dxc, "DxcCreateInstance"));
  if (create == nullptr) {
    Log.warn("dxcompiler.dll has no DxcCreateInstance");
    return false;
  }
  ComPtr<IDxcCompiler3Rt> compiler;
  HRESULT hr = create(kClsidDxcCompiler, kIidDxcCompiler3, reinterpret_cast<void**>(compiler.GetAddressOf()));
  if (FAILED(hr) || !compiler) {
    Log.warn("DXC compiler: {}", hresult(hr));
    return false;
  }
  const DxcBufferRt source{TraceSource, std::strlen(TraceSource), 65001 /* UTF-8 */};
  LPCWSTR args[] = {L"-T", L"cs_6_5", L"-E", L"main", L"-O3", L"-HV", L"2021"};
  ComPtr<IDxcOperationResultRt> result;
  hr = compiler->Compile(&source, args, static_cast<UINT32>(std::size(args)), nullptr, kIidDxcOperationResult,
                         reinterpret_cast<void**>(result.GetAddressOf()));
  if (FAILED(hr) || !result) {
    Log.warn("DXC compile call: {}", hresult(hr));
    return false;
  }
  HRESULT status = E_FAIL;
  result->GetStatus(&status);
  if (FAILED(status)) {
    ComPtr<IDxcBlobEncodingRt> errors;
    result->GetErrorBuffer(errors.GetAddressOf());
    std::string text;
    if (errors && errors->GetBufferSize() > 0) {
      text.assign(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
    }
    Log.warn("trace shader: {}", text);
    return false;
  }
  ComPtr<IDxcBlobRt> object;
  result->GetResult(object.GetAddressOf());
  if (!object || object->GetBufferSize() == 0) {
    Log.warn("trace shader: no object");
    return false;
  }
  hr = g_d3d.device->CreateRootSignature(0, object->GetBufferPointer(), object->GetBufferSize(),
                                         IID_PPV_ARGS(&rootSignature));
  if (FAILED(hr)) {
    Log.warn("trace root signature: {}", hresult(hr));
    return false;
  }
  D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
  desc.pRootSignature = rootSignature.Get();
  desc.CS.pShaderBytecode = object->GetBufferPointer();
  desc.CS.BytecodeLength = object->GetBufferSize();
  hr = g_d3d.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pipeline));
  if (FAILED(hr)) {
    Log.warn("trace pipeline: {}", hresult(hr));
    return false;
  }
  return true;
}

bool init_d3d12() {
  if (webgpu::g_backendType != wgpu::BackendType::D3D12) {
    Log.info("ray-traced shadows need the D3D12 backend");
    return false;
  }
  if (!g_device.HasFeature(wgpu::FeatureName::SharedTextureMemoryD3D12Resource)) {
    Log.warn("Dawn device lacks SharedTextureMemoryD3D12Resource");
    return false;
  }
  const ComPtr<ID3D12Device> base = dawn::native::d3d12::GetD3D12Device(g_device.Get());
  if (!base || FAILED(base.As(&g_d3d.device))) {
    Log.warn("no ID3D12Device5");
    return false;
  }
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5{};
  if (FAILED(g_d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5))) ||
      options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_1) {
    Log.info("the GPU has no DXR 1.1 (raytracing tier {})", static_cast<int>(options5.RaytracingTier));
    return false;
  }
  D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_5};
  if (FAILED(g_d3d.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) ||
      model.HighestShaderModel < D3D_SHADER_MODEL_6_5) {
    Log.info("the GPU has no shader model 6.5");
    return false;
  }
  g_d3d.queue = dawn::native::d3d12::GetD3D12CommandQueue(g_device.Get());
  if (!g_d3d.queue) {
    Log.warn("no D3D12 queue from Dawn");
    return false;
  }
  if (!compile_trace(g_d3d.rootSignature, g_d3d.pipeline)) {
    return false;
  }
  D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
  heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  heapDesc.NumDescriptors = 2;
  heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  if (FAILED(g_d3d.device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_d3d.heap)))) {
    Log.warn("descriptor heap");
    return false;
  }
  g_d3d.descriptorSize = g_d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  if (FAILED(g_d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_d3d.fence)))) {
    Log.warn("fence");
    return false;
  }
  g_d3d.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  for (auto& slot : g_d3d.slots) {
    if (FAILED(g_d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    IID_PPV_ARGS(&slot.allocator)))) {
      Log.warn("command allocator");
      return false;
    }
    slot.instances = make_buffer(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * MaxInstances, D3D12_HEAP_TYPE_UPLOAD,
                                 D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!slot.instances) {
      return false;
    }
  }
  ComPtr<ID3D12GraphicsCommandList> list;
  if (FAILED(g_d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_d3d.slots[0].allocator.Get(),
                                             nullptr, IID_PPV_ARGS(&list))) ||
      FAILED(list.As(&g_d3d.list))) {
    Log.warn("command list");
    return false;
  }
  g_d3d.list->Close();
  Log.info("ray-traced shadows ready (DXR 1.1)");
  return true;
}

// The shared textures at width x height: made in D3D12, imported into Dawn and kept in access.
bool ensure_shared(uint32_t width, uint32_t height) {
  if (g_render.depthTexture && g_render.width == width && g_render.height == height) {
    return true;
  }
  // The old ones may still be in flight: let the GPU finish with them first.
  wait_fence(g_d3d.fenceValue);
  for (auto* memory : {&g_render.depthMemory, &g_render.maskMemory}) {
    if (*memory) {
      wgpu::Texture& texture = memory == &g_render.depthMemory ? g_render.depthTexture : g_render.maskTexture;
      checked("RT shared texture end", [&] {
        wgpu::SharedTextureMemoryEndAccessState state{};
        memory->EndAccess(texture, &state);
      });
    }
  }
  g_render.depthMemory = {};
  g_render.depthTexture = {};
  g_render.depthView = {};
  g_render.maskMemory = {};
  g_render.maskTexture = {};
  g_render.maskView = {};
  g_render.traced = false;
  retire(g_d3d.depth);
  retire(g_d3d.mask);
  g_render.width = 0;
  g_render.height = 0;

  const auto make = [&](DXGI_FORMAT format, ComPtr<ID3D12Resource>& out) {
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
    // Simultaneous access: no layout to hand over between Dawn's command lists and ours (both on
    // Dawn's queue); each use promotes it from COMMON, and it decays back between submits.
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    const HRESULT hr = g_d3d.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&out));
    if (FAILED(hr)) {
      Log.warn("shared texture: {}", hresult(hr));
      return false;
    }
    return true;
  };
  if (!make(DXGI_FORMAT_R32_FLOAT, g_d3d.depth) || !make(DXGI_FORMAT_R16G16B16A16_FLOAT, g_d3d.mask)) {
    return false;
  }
  const auto import = [&](const ComPtr<ID3D12Resource>& resource, const char* label, wgpu::SharedTextureMemory& memory,
                          wgpu::Texture& texture, wgpu::TextureView& view) {
    return checked(label, [&] {
      dawn::native::d3d12::SharedTextureMemoryD3D12ResourceDescriptor d3dDescriptor;
      d3dDescriptor.resource = resource;
      wgpu::SharedTextureMemoryDescriptor descriptor{};
      descriptor.nextInChain = &d3dDescriptor;
      descriptor.label = label;
      memory = g_device.ImportSharedTextureMemory(&descriptor);
      texture = memory.CreateTexture();
      wgpu::SharedTextureMemoryBeginAccessDescriptor begin{};
      begin.initialized = true;
      memory.BeginAccess(texture, &begin);
      view = texture.CreateView();
    });
  };
  if (!import(g_d3d.depth, "RT Depth", g_render.depthMemory, g_render.depthTexture, g_render.depthView) ||
      !import(g_d3d.mask, "RT Shadow Mask", g_render.maskMemory, g_render.maskTexture, g_render.maskView)) {
    g_render.depthMemory = {};
    g_render.depthTexture = {};
    g_render.depthView = {};
    g_render.maskMemory = {};
    g_render.maskTexture = {};
    g_render.maskView = {};
    return false;
  }
  // The trace's view of them.
  D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_d3d.heap->GetCPUDescriptorHandleForHeapStart();
  D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
  srv.Format = DXGI_FORMAT_R32_FLOAT;
  srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv.Texture2D.MipLevels = 1;
  g_d3d.device->CreateShaderResourceView(g_d3d.depth.Get(), &srv, cpu);
  cpu.ptr += g_d3d.descriptorSize;
  D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
  uav.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  g_d3d.device->CreateUnorderedAccessView(g_d3d.mask.Get(), nullptr, &uav, cpu);
  g_render.width = width;
  g_render.height = height;
  return true;
}

// Builds the meshes the game sent since the last trace (their BLAS) and drops the forgotten.
void update_meshes(ID3D12GraphicsCommandList4* list, bool& built) {
  std::unordered_map<uint64_t, std::vector<float>> pending;
  std::vector<uint64_t> drop;
  {
    auto& s = shared();
    std::scoped_lock lock{s.mutex};
    pending.swap(s.pending);
    drop.swap(s.drop);
  }
  for (const uint64_t key : drop) {
    const auto it = g_d3d.meshes.find(key);
    if (it != g_d3d.meshes.end()) {
      retire(it->second.vertices);
      retire(it->second.blas);
      g_d3d.meshes.erase(it);
    }
  }
  for (auto& [key, positions] : pending) {
    const auto old = g_d3d.meshes.find(key);
    if (old != g_d3d.meshes.end()) {
      retire(old->second.vertices);
      retire(old->second.blas);
      g_d3d.meshes.erase(old);
    }
    const uint32_t triangles = static_cast<uint32_t>(positions.size() / 9);
    if (triangles == 0) {
      continue;
    }
    Mesh mesh;
    mesh.triangles = triangles;
    mesh.lastUsed = g_d3d.traces;
    const uint64_t bytes = static_cast<uint64_t>(triangles) * 9 * sizeof(float);
    mesh.vertices = make_buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                                D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!mesh.vertices) {
      continue;
    }
    void* mapped = nullptr;
    const D3D12_RANGE noRead{0, 0};
    if (FAILED(mesh.vertices->Map(0, &noRead, &mapped))) {
      continue;
    }
    std::memcpy(mapped, positions.data(), bytes);
    mesh.vertices->Unmap(0, nullptr);

    D3D12_RAYTRACING_GEOMETRY_DESC geometry{};
    geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometry.Triangles.VertexBuffer.StartAddress = mesh.vertices->GetGPUVirtualAddress();
    geometry.Triangles.VertexBuffer.StrideInBytes = 3 * sizeof(float);
    geometry.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometry.Triangles.VertexCount = triangles * 3;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    inputs.NumDescs = 1;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.pGeometryDescs = &geometry;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
    g_d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);
    if (info.ResultDataMaxSizeInBytes == 0) {
      continue;
    }
    mesh.blas = make_buffer(info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    ComPtr<ID3D12Resource> scratch = make_buffer(info.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!mesh.blas || !scratch) {
      continue;
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
    build.DestAccelerationStructureData = mesh.blas->GetGPUVirtualAddress();
    build.Inputs = inputs;
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    retire(scratch);
    g_d3d.meshes[key] = std::move(mesh);
    built = true;
    Log.info("mesh {:016x}: {} triangles, BLAS {} KiB", key, triangles, info.ResultDataMaxSizeInBytes / 1024);
  }
}

// The trace of the frame just submitted (render thread, after the submit).
bool trace(const Params& params) {
  collect_garbage();
  auto& slot = g_d3d.slots[g_d3d.slot];
  wait_fence(slot.fence);
  if (FAILED(slot.allocator->Reset()) || FAILED(g_d3d.list->Reset(slot.allocator.Get(), nullptr))) {
    return false;
  }
  ID3D12GraphicsCommandList4* list = g_d3d.list.Get();
  ++g_d3d.traces;
  bool built = false;
  update_meshes(list, built);
  if (built) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &barrier);
  }

  // The frame's areas, a BLAS each, as the top level.
  std::vector<uint64_t> scene;
  {
    auto& s = shared();
    std::scoped_lock lock{s.mutex};
    scene = s.scene;
  }
  std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances;
  for (const uint64_t key : scene) {
    const auto it = g_d3d.meshes.find(key);
    if (it == g_d3d.meshes.end() || !it->second.blas || instances.size() == MaxInstances) {
      continue;
    }
    it->second.lastUsed = g_d3d.traces;
    D3D12_RAYTRACING_INSTANCE_DESC instance{};
    instance.Transform[0][0] = 1.f;
    instance.Transform[1][1] = 1.f;
    instance.Transform[2][2] = 1.f;
    instance.InstanceMask = 0xFF;
    instance.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
    instance.AccelerationStructure = it->second.blas->GetGPUVirtualAddress();
    instances.push_back(instance);
  }
  // Meshes out of the scene for long go (the game re-sends one it needs again).
  std::vector<uint64_t> stale;
  for (const auto& [key, mesh] : g_d3d.meshes) {
    if (g_d3d.traces - mesh.lastUsed > 1800) {
      stale.push_back(key);
    }
  }
  if (!stale.empty()) {
    auto& s = shared();
    std::scoped_lock lock{s.mutex};
    for (const uint64_t key : stale) {
      s.known.erase(key);
      s.drop.push_back(key);
    }
  }
  const auto submit = [&] {
    ID3D12CommandList* lists[] = {list};
    g_d3d.queue->ExecuteCommandLists(1, lists);
    g_d3d.queue->Signal(g_d3d.fence.Get(), ++g_d3d.fenceValue);
    slot.fence = g_d3d.fenceValue;
    g_d3d.slot = (g_d3d.slot + 1) % Slots;
  };
  if (instances.empty()) {
    // Nothing to trace against yet; the builds still go.
    if (SUCCEEDED(list->Close()) && built) {
      submit();
    }
    return false;
  }
  void* mapped = nullptr;
  const D3D12_RANGE noRead{0, 0};
  if (FAILED(slot.instances->Map(0, &noRead, &mapped))) {
    if (SUCCEEDED(list->Close()) && built) {
      submit();
    }
    return false;
  }
  std::memcpy(mapped, instances.data(), instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
  slot.instances->Unmap(0, nullptr);

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
  inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
  inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
  inputs.NumDescs = static_cast<UINT>(instances.size());
  inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  inputs.InstanceDescs = slot.instances->GetGPUVirtualAddress();
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
  g_d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);
  if (!g_d3d.tlas || g_d3d.tlasSize < info.ResultDataMaxSizeInBytes) {
    retire(g_d3d.tlas);
    g_d3d.tlasSize = info.ResultDataMaxSizeInBytes * 2;
    g_d3d.tlas = make_buffer(g_d3d.tlasSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
  }
  if (!g_d3d.scratch || g_d3d.scratchSize < info.ScratchDataSizeInBytes) {
    retire(g_d3d.scratch);
    g_d3d.scratchSize = info.ScratchDataSizeInBytes * 2;
    g_d3d.scratch = make_buffer(g_d3d.scratchSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  }
  if (!g_d3d.tlas || !g_d3d.scratch) {
    if (SUCCEEDED(list->Close()) && built) {
      submit();
    }
    return false;
  }
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
  build.DestAccelerationStructureData = g_d3d.tlas->GetGPUVirtualAddress();
  build.Inputs = inputs;
  build.ScratchAccelerationStructureData = g_d3d.scratch->GetGPUVirtualAddress();
  list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
  {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &barrier);
  }

  float constants[28]{};
  std::memcpy(constants, params.viewToWorld, sizeof(params.viewToWorld));
  std::memcpy(constants + 12, params.frustum, sizeof(params.frustum));
  std::memcpy(constants + 16, params.sun, sizeof(params.sun));
  static const float sBias = [] {
    const char* env = std::getenv("MP_RT_BIAS");
    return env != nullptr ? static_cast<float>(std::atof(env)) : 0.04f;
  }();
  static const float sLength = [] {
    const char* env = std::getenv("MP_RT_LENGTH");
    return env != nullptr ? static_cast<float>(std::atof(env)) : 600.f;
  }();
  constants[20] = sBias;
  constants[21] = 0.0015f;
  constants[22] = sLength;
  list->SetComputeRootSignature(g_d3d.rootSignature.Get());
  ID3D12DescriptorHeap* heaps[] = {g_d3d.heap.Get()};
  list->SetDescriptorHeaps(1, heaps);
  list->SetPipelineState(g_d3d.pipeline.Get());
  list->SetComputeRoot32BitConstants(0, 28, constants, 0);
  list->SetComputeRootShaderResourceView(1, g_d3d.tlas->GetGPUVirtualAddress());
  list->SetComputeRootDescriptorTable(2, g_d3d.heap->GetGPUDescriptorHandleForHeapStart());
  list->Dispatch((g_render.width + Group - 1) / Group, (g_render.height + Group - 1) / Group, 1);
  if (FAILED(list->Close())) {
    return false;
  }
  submit();
  return true;
}

void shutdown_d3d12() {
  if (g_d3d.fence) {
    wait_fence(g_d3d.fenceValue);
  }
  if (g_d3d.event != nullptr) {
    CloseHandle(g_d3d.event);
  }
  g_d3d = {};
}
#else
bool init_d3d12() { return false; }
bool ensure_shared(uint32_t, uint32_t) { return false; }
bool trace(const Params&) { return false; }
void shutdown_d3d12() {}
#endif

void encode_begin(const wgpu::CommandEncoder& cmd, const Params& params) {
  const auto& layer = shadow::rt_layer_view();
  if (!layer) {
    return;
  }
  const bool draw = g_render.traced && g_render.depthView && g_render.maskView &&
                    params.misc[1] == g_render.width && params.misc[2] == g_render.height;
  g_render.traced = false;
  wgpu::BindGroup group;
  if (draw) {
    ensure_inject();
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .textureView = g_render.depthView},
        wgpu::BindGroupEntry{.binding = 1, .textureView = g_render.maskView},
    };
    const wgpu::BindGroupDescriptor groupDescriptor{
        .label = "RT Shadow Inject",
        .layout = g_render.injectLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    group = g_device.CreateBindGroup(&groupDescriptor);
  }
  // Cleared to 0 (nothing traced) unless there is a trace to write.
  const wgpu::RenderPassDepthStencilAttachment depth{
      .view = layer,
      .depthLoadOp = wgpu::LoadOp::Clear,
      .depthStoreOp = wgpu::StoreOp::Store,
      .depthClearValue = 0.f,
  };
  const wgpu::RenderPassDescriptor passDescriptor{
      .label = "RT Shadow Inject",
      .depthStencilAttachment = &depth,
      .timestampWrites = webgpu::gpu_prof::pass_writes("RT shadow inject"),
  };
  const auto pass = cmd.BeginRenderPass(&passDescriptor);
  if (draw) {
    pass.SetPipeline(g_render.inject);
    pass.SetBindGroup(0, group);
    pass.SetViewport(0.f, 0.f, static_cast<float>(g_render.width), static_cast<float>(g_render.height), 0.f, 1.f);
    pass.SetScissorRect(0, 0, g_render.width, g_render.height);
    pass.Draw(3);
  }
  pass.End();
}

void encode_trace(const EncoderTaskContext& ctx, const wgpu::CommandEncoder& cmd, const Params& params) {
  g_render.depthCopied = false;
  const auto& depth = webgpu::g_depthBuffer;
  const uint32_t width = params.misc[1];
  const uint32_t height = params.misc[2];
  if (width == 0 || height == 0 || !depth.view || g_failed.load()) {
    return;
  }
  if (!ensure_shared(width, height)) {
    disable("the textures shared with D3D12 could not be made");
    return;
  }
  const uint32_t samples = webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_graphicsConfig.msaaSamples : 1;
  ensure_copy(samples);
  ctx.queue.WriteBuffer(g_render.uniforms, 0, params.depth, sizeof(params.depth));
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = g_render.uniforms, .size = 16},
      wgpu::BindGroupEntry{.binding = 1, .textureView = depth.view},
      wgpu::BindGroupEntry{.binding = 2, .textureView = g_render.depthView},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "RT Depth Copy",
      .layout = g_render.copyLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto group = g_device.CreateBindGroup(&groupDescriptor);
  const wgpu::ComputePassDescriptor passDescriptor{.label = "RT Depth Copy"};
  const auto pass = cmd.BeginComputePass(&passDescriptor);
  pass.SetPipeline(g_render.copy);
  pass.SetBindGroup(0, group);
  pass.DispatchWorkgroups((width + Group - 1) / Group, (height + Group - 1) / Group, 1);
  pass.End();
  g_render.depthCopied = true;
}

void encode(const EncoderTaskContext& ctx, const wgpu::CommandEncoder& cmd, const void* payload, size_t payloadSize,
            void*) {
  if (payloadSize != sizeof(Params)) {
    return;
  }
  Params params;
  std::memcpy(&params, payload, sizeof(params));
  if (params.misc[0] == 0) {
    encode_begin(cmd, params);
  } else {
    encode_trace(ctx, cmd, params);
  }
}

void after_submit(const EncoderTaskCompletionContext&, const void* payload, size_t payloadSize, void*) {
  if (payloadSize != sizeof(Params)) {
    return;
  }
  Params params;
  std::memcpy(&params, payload, sizeof(params));
  if (params.misc[0] != 1 || !g_render.depthCopied) {
    return;
  }
  g_render.depthCopied = false;
  g_render.traced = trace(params);
}

// Rigid view -> world rows to world -> view rows.
void invert_rigid(const float in[3][4], float out[3][4]) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out[r][c] = in[c][r];
    }
    out[r][3] = -(in[0][r] * in[0][3] + in[1][r] * in[1][3] + in[2][r] * in[2][3]);
  }
}
} // namespace

bool available() {
  if (g_failed.load()) {
    return false;
  }
  static std::once_flag once;
  std::call_once(once, [] {
    if (const char* env = std::getenv("MP_RT_SHADOWS"); env != nullptr && std::atoi(env) == 0) {
      Log.info("ray-traced shadows off (MP_RT_SHADOWS=0)");
      g_failed.store(true);
      return;
    }
    if (!g_device) {
      return;
    }
    if (init_d3d12()) {
      g_ready.store(true);
    } else {
      g_failed.store(true);
    }
  });
  return g_ready.load();
}

void set_mesh(uint64_t key, const float* positions, uint32_t triangles) {
  auto& s = shared();
  std::scoped_lock lock{s.mutex};
  if (positions == nullptr || triangles == 0) {
    s.pending.erase(key);
    s.known.erase(key);
    s.drop.push_back(key);
    return;
  }
  s.pending[key].assign(positions, positions + static_cast<size_t>(triangles) * 9);
  s.known.insert(key);
}

uint32_t set_scene(const uint64_t* keys, uint32_t count) {
  auto& s = shared();
  std::scoped_lock lock{s.mutex};
  s.scene.assign(keys, keys + count);
  for (uint32_t i = 0; i < count; ++i) {
    if (!s.known.contains(keys[i])) {
      return i + 1;
    }
  }
  return 0;
}

bool ensure_task() {
  if (g_task == InvalidEncoderTask) {
    g_task = register_encoder_task_type(
        EncoderTaskDescriptor{.label = "RT Shadows", .callback = encode, .afterSubmit = after_submit});
    if (g_task == InvalidEncoderTask) {
      Log.warn("could not register the RT shadow task");
      return false;
    }
  }
  return true;
}

bool record(Params params, Uniform& out) {
  out = {};
  if (g_task == InvalidEncoderTask || !g_ready.load()) {
    g_fifo.traced = false;
    return false;
  }
  if (params.misc[0] == 0) {
    if (!g_fifo.traced) {
      return false;
    }
    g_fifo.traced = false;
    // This frame's view space -> world -> the traced frame's view space.
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        float sum = 0.f;
        for (int k = 0; k < 3; ++k) {
          sum += g_fifo.worldToView[r][k] * params.viewToWorld[k][c];
        }
        out.reproject[r][c] = sum + (c == 3 ? g_fifo.worldToView[r][3] : 0.f);
      }
    }
    std::memcpy(out.frustum, g_fifo.frustum, sizeof(out.frustum));
    params.misc[1] = g_fifo.width;
    params.misc[2] = g_fifo.height;
    if (!record_encoder_task(g_task, &params, sizeof(params))) {
      out = {};
      return false;
    }
    out.info[0] = 1.f;
    out.info[1] = static_cast<float>(g_fifo.width);
    out.info[2] = static_cast<float>(g_fifo.height);
    return true;
  }
  // The trace: the frame's size, as large as fits the shadow map's layer.
  const auto& size = webgpu::present_source().size;
  if (size.width == 0 || size.height == 0) {
    g_fifo.traced = false;
    return false;
  }
  const float layer = static_cast<float>(shadow::layer_size());
  const float scale = std::min({1.f, layer / static_cast<float>(size.width), layer / static_cast<float>(size.height)});
  const uint32_t width = std::max(1u, static_cast<uint32_t>(std::floor(static_cast<float>(size.width) * scale)));
  const uint32_t height = std::max(1u, static_cast<uint32_t>(std::floor(static_cast<float>(size.height) * scale)));
  params.misc[1] = width;
  params.misc[2] = height;
  if (!record_encoder_task(g_task, &params, sizeof(params))) {
    g_fifo.traced = false;
    return false;
  }
  g_fifo.traced = true;
  invert_rigid(params.viewToWorld, g_fifo.worldToView);
  std::memcpy(g_fifo.frustum, params.frustum, sizeof(g_fifo.frustum));
  g_fifo.width = width;
  g_fifo.height = height;
  return false;
}

void shutdown() {
  const auto task = g_task;
  g_task = InvalidEncoderTask;
  g_render = {};
  g_fifo = {};
  shutdown_d3d12();
  {
    auto& s = shared();
    std::scoped_lock lock{s.mutex};
    s.pending.clear();
    s.known.clear();
    s.scene.clear();
    s.drop.clear();
  }
  g_ready.store(false);
  if (task != InvalidEncoderTask) {
    unregister_encoder_task_type(task);
  }
}
} // namespace aurora::gfx::rt
