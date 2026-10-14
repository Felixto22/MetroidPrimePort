#pragma once

// Modern effects (port_debug.h ModernParticles, ModernRain, VolumetricFogAmount, VolumetricClouds,
// MotionBlur, DepthOfField, HeatDistortion, UnderwaterEffects, SnowFootprints, AmbientParticles):
//  - GPU particles (aurora/fxp.hpp) over the game's own: trails behind every beam and missile by
//    its type, bursts where they end, the charge beam gathering light at the muzzle, bubbles under
//    water, heat haze over lava and hot shots, footprints and morph ball tracks in snow;
//  - weather fields in place of the game's rain and snow (kept out from under roofs by a height
//    grid raycast around the camera, splashing on it), dust motes, embers and ash over lava;
//  - full-screen passes (GXPortPost): volumetric fog lit by the sun and the nearest lights,
//    caustics under water, wet and snowy ground, volumetric clouds over the sky; motion blur and
//    depth of field at the end of the world's draw.
#include "Kyoto/Math/CTransform4f.hpp"
#include "Kyoto/Math/CVector3f.hpp"

class CStateManager;

namespace PortFx {
// A fluid's surface drawn this frame (CScriptWater::Render): its CFluidPlane::EFluidType, box and
// surface height. The atmosphere reads the last frame's (it draws before the fluids).
void NoteFluid(int fluidType, const CVector3f& boxMin, const CVector3f& boxMax, float surfaceZ);
// Whether a world sky was drawn this frame (CWorld::DrawSky), for the clouds.
void NoteSky(bool drawn);
// Whether the game's own rain or snow (CEnvFxManager, EEnvFxType) is replaced by the GPU's.
bool ReplacesEnvFx(int envFxType);
// Whether the frame's scene capture is wanted (soft particles, the heat haze).
bool WantsSceneCapture();
// After the opaque world's SSAO and SSGI, before the scene capture: clouds, then the fog,
// caustics and wet or snowy ground. sunTravel: the sun light's direction (zero: no sun).
void DrawAtmosphere(const CStateManager& mgr, const CTransform4f& view, const float sunTravel[3]);
// After the world, the gun and the transparents: the particles and weather fields.
void DrawParticles(const CStateManager& mgr, const CTransform4f& view, bool sceneCaptured);
// After the particles, before the HUD: motion blur and depth of field.
void DrawFinish(const CStateManager& mgr, const CTransform4f& view);
// A health pack thrown from one point to another, landing in `seconds` (port_enemy_ai.h).
void ThrowPack(const CVector3f& from, const CVector3f& to, float seconds);
// A burst where the arm cannon struck (beam: CPlayerState::EBeamId).
void MeleeHit(const CVector3f& at, const CVector3f& dir, int beam);
// Forgets the particles and everything tracked (a world change).
void Reset();
} // namespace PortFx
