#pragma once

#include <cstdint>

#include <webgpu/webgpu_cpp.h>

// Port extension: ray-traced sun shadows (GXPortRtShadows) with DirectX Raytracing, beside WebGPU.
// Windows only, on Dawn's D3D12 device, with a DXR 1.1 GPU (inline ray queries); elsewhere
// available() is false and the sun keeps its shadow map.
//
// The game hands over the world's static opaque triangles once per area (set_mesh) and, each
// frame, which areas are drawn (set_scene); each becomes a bottom-level acceleration structure,
// and the frame's areas the top level. A frame then goes:
//  - Begin (before the world draws): the shadow traced after the previous frame, filtered, is
//    written into one more layer of the sun's shadow map (packed with its depth), and the
//    receivers' uniform gets the way from this frame's view to that one, so each receiver looks
//    its own point up in it, as the cascades are read a frame late too;
//  - Trace (after the opaque world): the world's linear depth is copied out to a texture shared
//    with D3D12, and once the frame is submitted a D3D12 compute shader on Dawn's own queue
//    reconstructs each texel's world point and casts rays toward the sun, within its disc, through
//    the acceleration structure.
// The world's opaque draws then stop casting into the shadow map (the actors and the alpha-tested
// world still do).
namespace aurora::gfx::rt {
// GXPortRtParams, word for word.
struct Params {
  float viewToWorld[3][4]; // GX's view space (right, up, towards the camera) to world, as rows
  float frustum[4];        // left, right, bottom, top at a view depth of 1
  float depth[4];          // near, far, and the GX z range the world draws in (min, max)
  float sun[4];            // world direction to the sun (xyz); w: tan of the sun disc's radius
  uint32_t misc[4];        // phase (0: begin, 1: trace), mask width, height, light slot (set here)
};
static_assert(sizeof(Params) == 28 * 4);

// The receivers' part of the shadow uniform (shadow::Uniform's tail).
struct Uniform {
  float reproject[3][4]; // this frame's view space to the traced frame's, as rows
  float frustum[4];      // the traced frame's frustum at a view depth of 1
  float info[4];         // 1 when there is a traced shadow, its width and height in texels, 0
};
static_assert(sizeof(Uniform) == 5 * 16);

// Whether the device can trace (any thread); the first call sets the D3D12 side up.
bool available();
// The world triangles (xyz of each corner, three corners a triangle) for key; null/0 forgets it.
void set_mesh(uint64_t key, const float* positions, uint32_t triangles);
// The frame's lights, world space: position, radius; colour, 0; GX attenuation k0, k1, k2, 0. Up
// to 64 (nearest the camera first); each casts shadows of the world onto what it lights.
void set_lights(const float (*lights)[12], uint32_t count);
// The last trace as the lit draws read it (any thread): r the sun's visibility, g the lights',
// b the texel's view depth (0: nothing traced there).
const wgpu::TextureView& mask_view();
// The areas to trace this frame; returns 1 + the index of the first key that has no mesh (pending
// or built), 0 when all do.
uint32_t set_scene(const uint64_t* keys, uint32_t count);
bool ensure_task();
// FIFO processor: a frame's phase. Begin fills out (and returns whether there is a traced shadow to
// read); Trace records the depth copy and the trace after the frame's submit.
bool record(Params params, Uniform& out);
void shutdown();
} // namespace aurora::gfx::rt
