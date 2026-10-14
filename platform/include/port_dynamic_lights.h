#pragma once

// Clustered lighting's extra lights (port_debug.h ClusteredLighting): every shot in flight glows
// on what it passes, in its beam's colour, and flashes where it ends. Retail lights only some
// of them, dimly; with the clustered lighting there's no limit on how many light a surface.
#include "Kyoto/Graphics/CLight.hpp"
#include "rstl/vector.hpp"

class CStateManager;

namespace PortDynamicLights {
// Adds the shots' lights and their impact flashes to the game's dynamic light list (the
// frame's PreRender).
void Append(const CStateManager& mgr, rstl::vector< CLight >& lights);
} // namespace PortDynamicLights
