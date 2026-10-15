#pragma once

#include "../webgpu/gpu.hpp"

#include <cstdint>

#include <webgpu/webgpu_cpp.h>

// Port extension: AMD FSR upscaling (FSR 4 on an RX 9000 or 7000, FSR 3.1 on other GPUs), from the
// FidelityFX SDK's signed DLLs (amd_fidelityfx_loader_dx12.dll and amd_fidelityfx_upscaler_dx12.dll
// beside the executable), run on Dawn's D3D12 queue (Windows, the D3D12 backend).
//  - The world is drawn at the render size (the frame buffer, scaled down by the mode's ratio)
//    with a sub-pixel jitter in its projection (GXPortSetJitter, from jitter());
//  - at the end of the world's draw (GXPortFsrCapture) the frame so far is copied: what the HUD
//    adds after it is told from the world by comparing the two;
//  - at the frame's end (upscale(), before the present) a compute pass makes FSR's inputs: the
//    colour, the depth, and motion vectors from the camera's motion since the last frame (the
//    world's position under each pixel taken to where the last view saw it; zero for the gun and
//    the HUD); the frame so far is submitted, FSR runs on the queue, and the present shows its
//    output at the display size.
namespace aurora::gfx::fsr {
enum Mode : uint32_t {
  Off = 0,
  NativeAA = 1,    // 1x: FSR as anti-aliasing
  Quality = 2,     // 1.5x
  Balanced = 3,    // 1.7x
  Performance = 4, // 2x
  ModeCount = 5,
};
// The per-dimension ratio of the display size to the render size.
float ratio(uint32_t mode);
// Whether FSR can run here (the D3D12 backend and the DLLs; checked once).
bool available();
// The jitter for frame `index` (pixels of the render size, each in -0.5..0.5).
void jitter(uint64_t index, uint32_t mode, float& x, float& y);

struct Params {
  float view[12];     // view to world, rows (GX view space: right, up, towards the camera; w the eye)
  float prevView[12]; // the same for the last frame
  float frustum[4];   // left, right, bottom, top at a view depth of 1
  float depth[4];     // near, far, and the GX z range the world draws in (min, max)
  float jitter[2];    // the frame's jitter (pixels of the render size)
  float frameMs;      // since the last frame
  float vfov;         // vertical field of view (radians)
  uint32_t mode;      // Mode
  uint32_t reset;     // the camera cut: no history
  uint32_t pad[2];
};
static_assert(sizeof(Params) == 40 * 4);

// Registers the capture task (game thread); false if it could not be.
bool ensure_task();
// From the FIFO processor (GX_AURORA_PORT_FSR): the world's picture is complete.
bool record(const Params& params);
// Render worker, at the frame's end before the present: when this frame was captured, makes the
// inputs, submits `encoder` (and replaces it with a fresh one) and upscales. Returns the picture
// to present instead of the frame buffer (nullptr: present the frame buffer).
const webgpu::TextureWithSampler* upscale(wgpu::CommandEncoder& encoder);
void shutdown();
} // namespace aurora::gfx::fsr
