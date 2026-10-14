#pragma once

#include <cstdint>

#include "Kyoto/Graphics/CLight.hpp"
#include "Kyoto/Math/CVector3f.hpp"
#include "rstl/vector.hpp"

class CStateManager;

// Cinematic look (port_debug.h CinematicLook): on the retail picture (rooms without Remastered's
// tone curve), a bloom that lets the brightest light (shots, explosions, the sky, waterfalls,
// glowing creatures) glow over what surrounds it, and a filmic colour grade (more contrast and
// colour, cool shadows and warm highlights), through GXPortPostProcess.
namespace PortCinematic {
// The post-process for a retail frame: false when the look is off.
// Also hands GXPortPostProcess its extras: the lens star and the light shafts from the sun.
bool Retail(float& threshold, float tints[5][3], float tone[3][4], uint32_t& grade);
// The room's sun, the way its light travels (zero: none), from the shadow setup.
void SetSun(const float dir[3]);
// Once a frame (the dynamic light list's build): the cannon's heat from the shots Samus fires,
// cooling between them; adds the hot muzzle's glow to `lights`.
void TickGun(const CStateManager& mgr, rstl::vector< CLight >& lights);
// The muzzle about to draw (world point, under the gun's view): its heat glow.
void SetGunHeatAt(const CVector3f& muzzleWorld);
// The smoke off the hot muzzle, in the gun's own space (where the muzzle is `muzzleLocal`).
void RenderGunSmoke(const CVector3f& muzzleLocal);
// The arm cannon's metal highlights, around its draw.
struct GunShine {
  GunShine();
  ~GunShine();
  bool on;
};
} // namespace PortCinematic
