#include "port_foliage.h"

#include "port_debug.h"

#include "Kyoto/Graphics/CGraphics.hpp"
#include "Kyoto/Math/CTransform4f.hpp"
#include "Kyoto/Math/CVector3f.hpp"
#include "MetroidPrime/CObjectList.hpp"
#include "MetroidPrime/CStateManager.hpp"
#include "MetroidPrime/Player/CPlayer.hpp"
#include "MetroidPrime/TCastTo.hpp"
#include "MetroidPrime/Weapons/CGameProjectile.hpp"

#include <dolphin/gx/GXExtra.h>

#include <algorithm>
#include <cmath>
#include <iterator>

namespace PortFoliage {
namespace {
// Plant textures (TXTR ids), picked by eye from the disc's cut-out textures: ferns, leaf sprays,
// bushes, grass fringes, hanging vines, roots and moss. Wall and floor decals (moss patches,
// debris) stay off the list: bending a decal off its wall would show the gap behind it.
const unsigned int kFoliageTextures[] = {
    0x0AF0DFD0, 0x10C5E944, 0x12ACA553, 0x1AB3E109, 0x1B7FFC3F, 0x1D19EFF8,
    0x22FC069C, 0x244C952A, 0x2520FD93, 0x280B89C9, 0x2A9AF9B0, 0x2AB88EC6,
    0x2DCF3C32, 0x3335C23F, 0x3F578E77, 0x426A772D, 0x45D5C2E1, 0x46D630EF,
    0x4CF9A85C, 0x50CFA33D, 0x526C63ED, 0x56954560, 0x5695CA37, 0x59EB930D,
    0x6114F166, 0x64A444DB, 0x6B211DB9, 0x72852E14, 0x779CEE5C, 0x81EA28CF,
    0x8DFB0DC8, 0x90DE0FF1, 0x91C5E27D, 0x96939AD5, 0x9DB417D9, 0xA670D1C9,
    0xA9374EE7, 0xA9E79300, 0xB2D2A624, 0xB3936FB4, 0xB782B593, 0xBD05E899,
    0xBDA1ACFB, 0xBE16E803, 0xC694D876, 0xC7D58962, 0xD9A89D7B, 0xE0548F3C,
    0xE2DD22D8, 0xF0910EF1, 0xF5963EE5, 0xFE0E9414, 0xFF2252D5, 0xFF9FA3CD,
};

struct Pusher {
  CVector3f pos = CVector3f::Zero(); // feet
  float radius = 0.f;
  float height = 0.f;
  float strength = 0.f;
  float age = 0.f; // seconds since it let go
};

// Released pushers: where Samus or a shot was, swinging the foliage back.
const int kMaxReleased = 24;
Pusher sReleased[kMaxReleased];
int sReleasedCount = 0;

// Live shots seen last frame, to leave a trail and a burst where they end.
struct Shot {
  TUniqueId uid = kInvalidUniqueId;
  CVector3f last = CVector3f::Zero();
  CVector3f trail = CVector3f::Zero();
  bool seen = false;
};
const int kMaxShots = 16;
Shot sShots[kMaxShots];
int sShotCount = 0;

bool sHavePlayer = false;
CVector3f sPlayerTrail = CVector3f::Zero();
float sLastClock = -1.f;
bool sSentEmpty = false;

const float kReleaseLife = 1.3f;
const float kPlayerTrailStep = 0.8f;
const float kShotTrailStep = 1.0f;

void Release(const CVector3f& pos, float radius, float height, float strength) {
  if (sReleasedCount == kMaxReleased) {
    // Drop the oldest.
    int oldest = 0;
    for (int i = 1; i < sReleasedCount; ++i) {
      if (sReleased[i].age > sReleased[oldest].age) {
        oldest = i;
      }
    }
    sReleased[oldest] = sReleased[--sReleasedCount];
  }
  Pusher& p = sReleased[sReleasedCount++];
  p.pos = pos;
  p.radius = radius;
  p.height = height;
  p.strength = strength;
  p.age = 1e-3f;
}

float Advance() {
  const float clock = CGraphics::GetSecondsMod900();
  float dt = sLastClock < 0.f ? 0.f : clock - sLastClock;
  if (dt < 0.f) {
    dt += 900.f;
  }
  sLastClock = clock;
  return std::clamp(dt, 0.f, 0.1f);
}
} // namespace

bool IsFoliageTexture(unsigned int assetId) {
  return std::binary_search(std::begin(kFoliageTextures), std::end(kFoliageTextures), assetId);
}

void Upload(const CStateManager& mgr) {
  if (!PortDebug::InteractiveFoliage()) {
    if (!sSentEmpty) {
      const f32 up[3] = {0.f, 1.f, 0.f};
      GXPortSetFoliagePushers(up, nullptr, nullptr, 0);
      sSentEmpty = true;
    }
    sReleasedCount = 0;
    sShotCount = 0;
    sHavePlayer = false;
    sLastClock = -1.f;
    return;
  }
  sSentEmpty = false;

  const float dt = Advance();
  for (int i = sReleasedCount - 1; i >= 0; --i) {
    sReleased[i].age += dt;
    if (sReleased[i].age > kReleaseLife) {
      sReleased[i] = sReleased[--sReleasedCount];
    }
  }

  Pusher live[GX_PORT_FOLIAGE_MAX_PUSHERS];
  int liveCount = 0;

  // Samus: her body, or the morph ball.
  if (const CPlayer* player = mgr.GetPlayer()) {
    const CVector3f feet = player->GetTranslation();
    const bool ball = player->GetMorphballTransitionState() == CPlayer::kMS_Morphed;
    const float radius = ball ? 0.8f : 0.75f;
    const float height = ball ? 0.9f : 2.6f;
    if (!sHavePlayer || (feet - sPlayerTrail).Magnitude() > 10.f) {
      sPlayerTrail = feet;
      sHavePlayer = true;
    } else if ((feet - sPlayerTrail).Magnitude() > kPlayerTrailStep) {
      Release(sPlayerTrail, radius, height, 1.f);
      sPlayerTrail = feet;
    }
    Pusher& p = live[liveCount++];
    p.pos = feet;
    p.radius = radius;
    p.height = height;
    p.strength = 1.f;
  }

  // Shots in flight push what they pass through; where one ends, a burst.
  const CVector3f eye = CGraphics::GetViewMatrix().GetTranslation();
  for (int i = 0; i < sShotCount; ++i) {
    sShots[i].seen = false;
  }
  const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
  for (int idx = actors.GetFirstObjectIndex(); idx != -1; idx = actors.GetNextObjectIndex(idx)) {
    const CGameProjectile* proj = TCastToConstPtr< CGameProjectile >(actors.GetObjectByIndex(idx));
    if (proj == nullptr || !proj->GetActive()) {
      continue;
    }
    const CVector3f pos = proj->GetTranslation();
    if ((pos - eye).Magnitude() > 40.f) {
      continue;
    }
    const CVector3f feet = pos - CVector3f(0.f, 0.f, 0.6f);
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
      shot->trail = feet;
    }
    if (shot != nullptr) {
      shot->seen = true;
      shot->last = feet;
      if ((feet - shot->trail).Magnitude() > kShotTrailStep) {
        Release(shot->trail, 0.8f, 1.2f, 0.8f);
        shot->trail = feet;
      }
    }
    if (liveCount < 6) {
      Pusher& p = live[liveCount++];
      p.pos = feet;
      p.radius = 0.8f;
      p.height = 1.2f;
      p.strength = 0.8f;
    }
  }
  for (int i = sShotCount - 1; i >= 0; --i) {
    if (!sShots[i].seen) {
      Release(sShots[i].last, 1.3f, 1.6f, 1.2f);
      sShots[i] = sShots[--sShotCount];
    }
  }

  // Live pushers first, then the freshest of the released ones.
  Pusher* released[kMaxReleased];
  for (int i = 0; i < sReleasedCount; ++i) {
    released[i] = &sReleased[i];
  }
  std::sort(released, released + sReleasedCount,
            [](const Pusher* a, const Pusher* b) { return a->age < b->age; });

  // GX's view space: the camera's right, up and backward rows, about its eye.
  const CTransform4f& view = CGraphics::GetViewMatrix();
  const float axes[3][3] = {
      {view.Get00(), view.Get10(), view.Get20()},
      {view.Get02(), view.Get12(), view.Get22()},
      {-view.Get01(), -view.Get11(), -view.Get21()},
  };
  const auto toView = [&](const CVector3f& w, f32 out[4]) {
    const float d[3] = {w.GetX() - eye.GetX(), w.GetY() - eye.GetY(), w.GetZ() - eye.GetZ()};
    for (int r = 0; r < 3; ++r) {
      out[r] = axes[r][0] * d[0] + axes[r][1] * d[1] + axes[r][2] * d[2];
    }
  };
  const f32 up[3] = {axes[0][2], axes[1][2], axes[2][2]};
  f32 pos[GX_PORT_FOLIAGE_MAX_PUSHERS][4];
  f32 params[GX_PORT_FOLIAGE_MAX_PUSHERS][4];
  u32 count = 0;
  const auto add = [&](const Pusher& p, float age) {
    toView(p.pos, pos[count]);
    pos[count][3] = p.radius;
    params[count][0] = p.strength;
    params[count][1] = age;
    params[count][2] = p.height;
    params[count][3] = 0.f;
    ++count;
  };
  for (int i = 0; i < liveCount && count < GX_PORT_FOLIAGE_MAX_PUSHERS; ++i) {
    add(live[i], 0.f);
  }
  for (int i = 0; i < sReleasedCount && count < GX_PORT_FOLIAGE_MAX_PUSHERS; ++i) {
    add(*released[i], released[i]->age);
  }
  GXPortSetFoliagePushers(up, pos, params, count);
}
} // namespace PortFoliage
