// Ray-traced sun shadows (port_rt_shadows.h).
#include "port_rt_shadows.h"

#include "port_debug.h"
#include "port_env.h"

#include "Kyoto/Graphics/CGraphics.hpp"
#include "MetaRender/CCubeRenderer.hpp"
#include "MetroidPrime/CGameArea.hpp"

#include <dolphin/gx/GXExtra.h>

#include <cmath>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace PortRtShadows {
namespace {
bool sActive = false;
GXPortRtParams sParams;
// Areas whose geometry had no opaque triangle (nothing to send, nothing to ask again).
std::unordered_set< u64 > sEmpty;
std::vector< float > sTriangles;

u64 AreaKey(const CGameArea& area) {
  // The area's MREA: the same geometry whenever it's loaded, in whichever world.
  return (u64(1) << 63) | u64(area.GetAreaAssetId());
}
} // namespace

bool Begin(const CGameArea* const* areas, int areaCount, const CTransform4f& view, const float sunTravel[3]) {
  sActive = false;
  const float len = std::sqrt(sunTravel[0] * sunTravel[0] + sunTravel[1] * sunTravel[1] + sunTravel[2] * sunTravel[2]);
  if (!PortDebug::RayTracedShadows() || !(len > 0.f) || areaCount <= 0 || !GXPortRtAvailable()) {
    return false;
  }
  u64 keys[16];
  int count = 0;
  const CGameArea* keyAreas[16];
  for (int i = 0; i < areaCount && count < 16; ++i) {
    if (areas[i] == nullptr) {
      continue;
    }
    const u64 key = AreaKey(*areas[i]);
    if (sEmpty.count(key) != 0) {
      continue;
    }
    keyAreas[count] = areas[i];
    keys[count++] = key;
  }
  if (count == 0) {
    return false;
  }
  // One area's geometry a frame (the walk over its display lists takes a moment).
  const u32 missing = GXPortRtScene(keys, static_cast< u32 >(count));
  if (missing != 0) {
    const CGameArea& area = *keyAreas[missing - 1];
    if (gpRender != nullptr && gpRender->PortAreaTriangles(area.GetId().Value(), sTriangles) &&
        sTriangles.size() >= 9) {
      GXPortRtMesh(keys[missing - 1], sTriangles.data(), static_cast< u32 >(sTriangles.size() / 9));
    } else {
      sEmpty.insert(keys[missing - 1]);
    }
    sTriangles.clear();
    sTriangles.shrink_to_fit();
  }

  memset(&sParams, 0, sizeof(sParams));
  // View space is GX's: right, up, towards the camera.
  const CVector3f eye = view.GetTranslation();
  const float rows[3][4] = {
      {view.Get00(), view.Get02(), -view.Get01(), eye.GetX()},
      {view.Get10(), view.Get12(), -view.Get11(), eye.GetY()},
      {view.Get20(), view.Get22(), -view.Get21(), eye.GetZ()},
  };
  memcpy(sParams.viewToWorld, rows, sizeof(rows));
  const CGraphics::CProjectionState& proj = CGraphics::GetProjectionState();
  const float zNear = proj.GetNear();
  sParams.frustum[0] = proj.GetLeft() / zNear;
  sParams.frustum[1] = proj.GetRight() / zNear;
  sParams.frustum[2] = proj.GetBottom() / zNear;
  sParams.frustum[3] = proj.GetTop() / zNear;
  sParams.depth[0] = zNear;
  sParams.depth[1] = proj.GetFar();
  // SetupViewForDraw's depth range, as the volumetric fog reads it.
  sParams.depth[2] = 0.125f;
  sParams.depth[3] = 1.f;
  for (int i = 0; i < 3; ++i) {
    sParams.sun[i] = -sunTravel[i] / len;
  }
  // The sun disc's radius (tan): the penumbra widens with the distance to what casts it.
  static const float sSoftness = port::EnvFloat("MP_RT_SOFTNESS", 0.02f);
  sParams.sun[3] = sSoftness;
  sParams.misc[0] = 0;
  if (!GXPortRtShadows(&sParams)) {
    return false;
  }
  sActive = true;
  return true;
}

void Trace() {
  if (!sActive) {
    return;
  }
  sActive = false;
  sParams.misc[0] = 1;
  GXPortRtShadows(&sParams);
}
} // namespace PortRtShadows
