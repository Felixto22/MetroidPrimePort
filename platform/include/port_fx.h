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

#include <cstdint>

class CStateManager;
class CActor;

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
// Outdoors on the planet with the day and night on: the new sky and its clouds drawn first, over
// the whole view, for the world to draw over (retail's skies and cloud domes are left out).
bool DrawSkyFirst(const CStateManager& mgr, const CTransform4f& view, const float sunTravel[3]);
bool SkyFirstDrawn();
// The sky's ambient on the world lit live outdoors (GXPortSetRelightAmbient); 0 without the new sky.
float WorldAmbient();
// The day and night's parts, for finding which one misdraws (F1 > Day and night > Parts).
enum DiagBits : uint32_t {
  kDiagSkyFirst = 1,   // the new sky drawn first, behind the world
  kDiagHideDomes = 2,  // retail's cloud layers left out
  kDiagKeyLight = 4,   // the sun or moon drives the shadows and the materials' sun
  kDiagRelight = 8,    // lightmaps off, the world lit live
  kDiagGrade = 16,     // the time of day's colour over the world
  kDiagFogTint = 32,   // the fog lit by the sun or moon
  kDiagMagenta = 64,   // debug: the new sky drawn magenta
  kDiagDefault = 63,
};
uint32_t Diag();
void SetDiag(uint32_t bits);
// After the world, the gun and the transparents: the particles and weather fields.
void DrawParticles(const CStateManager& mgr, const CTransform4f& view, bool sceneCaptured);
// After the particles, before the HUD: motion blur and depth of field.
void DrawFinish(const CStateManager& mgr, const CTransform4f& view);
// FSR (port_debug.h Upscaling): the jitter on at the start of the world's draw (a normal, not
// zoomed, pass), and off with the capture of the finished world at its end (before the HUD).
void FsrBegin();
void FsrEnd(const CTransform4f& view);
// The opaque world is drawn (before the actors and effects): FSR's reactive mask is what they add,
// so explosions, fire and beams don't linger in its history.
void FsrSolid();
// Flames at a point of a burning plant this frame (strength 0..1, dt: the frame's step).
void Fire(const CVector3f& at, float strength, float dt);
// The lightning's flash (0..1), for what else it lights.
float LightningFlash();
// Day and night, and the weather (port_debug.h DayNight, DynamicWeather), outdoors (where the
// world's sky was drawn last frame). Once a frame: the clock and the weather move on.
void SkyTick(const CStateManager& mgr);
// With the day and night on the planet, retail's sky domes (Tallon's) give way to the sky drawn
// for the hour: true for one to leave out of the frame.
bool ReplacesSkyDome(const CActor& actor);
// Outdoors with the day and night on: the way the key light travels (the sun by day, the moon by
// night), its intensity (0..1) and colour. False: the room's own sun stands.
bool SkyLight(float travel[3], float& intensity, float color[3]);
// The time of day in hours (0..24).
float DayHour();
// A health pack thrown from one point to another, landing in `seconds` (port_enemy_ai.h).
void ThrowPack(const CVector3f& from, const CVector3f& to, float seconds);
// A burst where the arm cannon struck (beam: CPlayerState::EBeamId).
void MeleeHit(const CVector3f& at, const CVector3f& dir, int beam);
// A pirate grenade in flight (each game tick): its glow, blinking faster as the fuse runs down.
void GrenadeTick(const CVector3f& at, float fuse, bool returned);
// A grenade going off.
void Explosion(const CStateManager& mgr, const CVector3f& at, float scale);
// The cannon strike breaking through a shield, and shattering a frozen enemy.
void ShieldBreak(const CVector3f& at, const CVector3f& dir);
void Shatter(const CVector3f& at);
// A pirate enraged (its partner died): the burst, and the aura it keeps.
void Rage(const CVector3f& at);
void RageAura(const CVector3f& at);
// Forgets the particles and everything tracked (a world change).
void Reset();
} // namespace PortFx
