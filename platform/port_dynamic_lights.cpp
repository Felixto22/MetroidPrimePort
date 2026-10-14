// Shot and impact lights for the clustered lighting (port_dynamic_lights.h).
#include "port_dynamic_lights.h"

#include "port_debug.h"

#include "Kyoto/Graphics/CColor.hpp"
#include "Kyoto/Graphics/CGraphics.hpp"
#include "Kyoto/Math/CVector3f.hpp"
#include "MetroidPrime/CObjectList.hpp"
#include "MetroidPrime/CStateManager.hpp"
#include "MetroidPrime/Player/CPlayer.hpp"
#include "MetroidPrime/TCastTo.hpp"
#include "MetroidPrime/Weapons/CGameProjectile.hpp"

#include <algorithm>
#include <cmath>

namespace PortDynamicLights {
namespace {
struct Shot {
  TUniqueId uid = kInvalidUniqueId;
  CVector3f pos = CVector3f::Zero();
  float rgb[3] = {};
  bool seen = false;
};
const int kMaxShots = 48;
Shot sShots[kMaxShots];
int sShotCount = 0;

struct Flash {
  CVector3f pos = CVector3f::Zero();
  float rgb[3] = {};
  float age = 0.f;
};
const int kMaxFlashes = 24;
Flash sFlashes[kMaxFlashes];
int sFlashCount = 0;
const float kFlashLife = 0.45f;

float sLastClock = -1.f;
const int kIdBase = 0x53484f54; // 'SHOT': apart from the game's light ids

// A beam's glow colour.
void ColorFor(EWeaponType type, bool player, float out[3]) {
  float r = 1.f, g = 0.55f, b = 0.25f; // enemy shots: hot orange
  if (player) {
    switch (type) {
    case kWT_Power:
      r = 1.f, g = 0.8f, b = 0.35f;
      break;
    case kWT_Ice:
      r = 0.45f, g = 0.8f, b = 1.f;
      break;
    case kWT_Wave:
      r = 0.75f, g = 0.4f, b = 1.f;
      break;
    case kWT_Plasma:
      r = 1.f, g = 0.4f, b = 0.15f;
      break;
    case kWT_Missile:
      r = 1.f, g = 0.6f, b = 0.3f;
      break;
    case kWT_Phazon:
      r = 0.3f, g = 0.6f, b = 1.f;
      break;
    default:
      break;
    }
  }
  out[0] = r, out[1] = g, out[2] = b;
}

CLight MakeLight(const CVector3f& pos, const float rgb[3], float intensity, float quadratic, int id) {
  CLight light = CLight::BuildPoint(pos, CColor(rgb[0] * intensity, rgb[1] * intensity, rgb[2] * intensity, 1.f));
  light.SetAttenuation(1.f, 0.f, quadratic);
  light.SetId(id);
  return light;
}
} // namespace

void Append(const CStateManager& mgr, rstl::vector< CLight >& lights) {
  if (!PortDebug::ClusteredLighting()) {
    sShotCount = 0;
    sFlashCount = 0;
    sLastClock = -1.f;
    return;
  }
  const float clock = CGraphics::GetSecondsMod900();
  float dt = sLastClock < 0.f ? 0.f : clock - sLastClock;
  if (dt < 0.f) {
    dt += 900.f;
  }
  dt = std::clamp(dt, 0.f, 0.1f);
  sLastClock = clock;

  for (int i = sFlashCount - 1; i >= 0; --i) {
    sFlashes[i].age += dt;
    if (sFlashes[i].age >= kFlashLife) {
      sFlashes[i] = sFlashes[--sFlashCount];
    }
  }
  for (int i = 0; i < sShotCount; ++i) {
    sShots[i].seen = false;
  }

  const int gameLights = lights.size();
  const CPlayer* player = mgr.GetPlayer();
  int id = kIdBase;
  const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
  for (int idx = actors.GetFirstObjectIndex(); idx != -1; idx = actors.GetNextObjectIndex(idx)) {
    const CGameProjectile* proj = TCastToConstPtr< CGameProjectile >(actors.GetObjectByIndex(idx));
    if (proj == nullptr || !proj->GetActive()) {
      continue;
    }
    const CVector3f pos = proj->GetTranslation();
    Shot* shot = nullptr;
    for (int i = 0; i < sShotCount; ++i) {
      if (sShots[i].uid == proj->GetUniqueId()) {
        shot = &sShots[i];
        break;
      }
    }
    if (shot == nullptr && sShotCount < kMaxShots) {
      shot = &sShots[sShotCount++];
      shot->uid = proj->GetUniqueId();
      const bool byPlayer = player != nullptr && proj->GetOwnerId() == player->GetUniqueId();
      ColorFor(proj->GetType(), byPlayer, shot->rgb);
    }
    if (shot == nullptr) {
      continue;
    }
    shot->seen = true;
    shot->pos = pos;
    // A shot the game already lights keeps its own light.
    bool lit = false;
    for (int i = 0; i < gameLights && !lit; ++i) {
      lit = (lights[i].GetPosition() - pos).MagSquared() < 1.f;
    }
    if (!lit) {
      lights.push_back(MakeLight(pos, shot->rgb, 0.9f, 0.3f, id++));
    }
  }
  // A shot gone this frame hit something: a flash where it was.
  for (int i = sShotCount - 1; i >= 0; --i) {
    if (!sShots[i].seen) {
      if (sFlashCount < kMaxFlashes) {
        Flash& f = sFlashes[sFlashCount++];
        f.pos = sShots[i].pos;
        std::copy(sShots[i].rgb, sShots[i].rgb + 3, f.rgb);
        f.age = 0.f;
      }
      sShots[i] = sShots[--sShotCount];
    }
  }
  for (int i = 0; i < sFlashCount; ++i) {
    const float k = 1.f - sFlashes[i].age / kFlashLife;
    lights.push_back(MakeLight(sFlashes[i].pos, sFlashes[i].rgb, 1.8f * k * k, 0.18f, id++));
  }
}
} // namespace PortDynamicLights
