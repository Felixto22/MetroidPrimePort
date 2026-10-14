#pragma once

// GPU-built geometry (aurora/geo.hpp, port_debug.h GpuGrass, Relief, Debris): grass over the
// grassy ground of the drawn areas, chunks of rock knocked loose where shots hit the world, and
// the relief strength of the Remastered surfaces.
#include "Kyoto/Math/CTransform4f.hpp"

class CGameArea;
class CStateManager;

namespace PortGpuGeo {
// After the areas' opaque world is drawn (normal visor): the grass of each area, the debris.
void DrawWorld(const CStateManager& mgr, const CGameArea* const* areas, int areaCount, const CTransform4f& view);
// Forgets every area's grass and the debris (a world change).
void Reset();
} // namespace PortGpuGeo
