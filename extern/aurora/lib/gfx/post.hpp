#pragma once

#include <cstdint>

// Port extension: full-screen passes over the frame as drawn so far (see GXPortPost), each a copy
// of the frame and the depth buffer read by one fragment pass that writes every pixel back:
//  - Atmosphere: volumetric fog marched from the eye to each pixel (height fog with drifting
//    noise, lit by the room's ambient, the sun through a phase function and up to eight point
//    lights scattering in it); caustics on the surfaces under a water plane (or everywhere when
//    the eye is under water); surfaces turned up to a rain darkened and rippled.
//  - Clouds: on the sky's pixels, a layer of clouds marched through (procedural noise drifting
//    with the wind, lit towards the sun).
//  - Finish: motion blur from the camera's motion since the last frame (each pixel's world
//    position taken to where the last frame's view saw it), and depth of field around a focus
//    distance (a gather over a disc as wide as the pixel's circle of confusion).
// The viewmodel (nearer than the world's depth range) is left alone by all three. The frame holds
// the tone-mapped colour gamma encoded: the passes work in linear and encode again.
namespace aurora::gfx::post {
enum Mode : uint32_t {
  Atmosphere = 0,
  Clouds = 1,
  Finish = 2,
  ModeCount = 3,
};

// The task's uniform, word for word (GXPortPostParams).
struct Params {
  float frustum[4];     // left, right, bottom, top at a view depth of 1
  float depth[4];       // near, far, and the GX z range the world draws in (min, max)
  float view[12];       // view to world, rows (GX view space: right, up, towards the camera; w the eye)
  float prevView[12];   // the same for the last frame
  float fog[4];         // density (1/m), height falloff (1/m), base height (world z), noise (0..1)
  float fogColor[4];    // ambient in-scatter (linear rgb), anisotropy (-1..1)
  float sun[4];         // direction to the sun (world), intensity (0: none)
  float sunColor[4];    // linear rgb, max fog distance (m)
  float lights[8][8];   // position xyz, radius; colour rgb (linear), intensity
  float water[4];       // surface height, caustics strength, eye under water (0/1), depth they reach (m)
  float waterBox[4];    // the surface's min x, min y, max x, max y
  float weather[4];     // wetness (0..1), time (s), snow cover (0..1), 0
  float clouds[4];      // coverage (0..1), base height above the eye (m), thickness (m), wind speed (m/s)
  float cloudColor[4];  // lit colour (linear rgb), density
  float finish[4];      // motion blur (0..1), depth of field (0..1), focus distance (m; negative: the screen centre's), focus range (m)
  uint32_t misc[4];     // mode, light count, frame counter, time slowed (0..1000; finish)
};
static_assert(sizeof(Params) == 140 * 4);

// Registers the task (game thread); false if it could not be.
bool ensure_task();
// Records the pass from the FIFO processor (GX_AURORA_PORT_POST), once ensure_task has returned true.
bool record(const Params& params);
void shutdown();
} // namespace aurora::gfx::post
