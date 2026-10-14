#pragma once

// Ray-traced sun shadows (port_debug.h RayTracedShadows): the world's static opaque geometry goes
// to aurora's DirectX Raytracing pass (aurora::gfx::rt), which traces the sun's shadow of it for
// every pixel; the shadow map then only takes the actors and the alpha-tested world.
#include "Kyoto/Math/CTransform4f.hpp"

class CGameArea;

namespace PortRtShadows {
// Before the world draws, after the frame's shadow setup: the areas drawn this frame, the camera
// and the way the sun's light travels (world space). True when the opaque world's shadow is traced
// this frame (it then leaves the shadow map to the rest).
bool Begin(const CGameArea* const* areas, int areaCount, const CTransform4f& view, const float sunTravel[3]);
// Once the opaque world is drawn: its depth is traced after the frame.
void Trace();
} // namespace PortRtShadows
