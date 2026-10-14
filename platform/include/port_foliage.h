#pragma once

// Interactive foliage (port_debug.h InteractiveFoliage, RainDetails, BurnableFoliage): plants bend
// away from Samus and from shots passing through them and sway back after; rain wets them and
// splashes on them; a beam shot into one sets it alight.
#include "Kyoto/Graphics/CLight.hpp"
#include "rstl/vector.hpp"

class CCubeModel;
class CCubeSurface;
class CStateManager;

namespace PortFoliage {
// Whether a texture (TXTR asset id) is a plant's: leaves, ferns, grass, hanging vines and moss.
bool IsFoliageTexture(unsigned int assetId);
// Whether any of the foliage settings is on (materials are flagged as foliage only then).
bool Active();
// A foliage surface about to draw, with the model matrix it draws under: remembered (its world
// bounds and some of its vertices) for the rain to splash on and the shots to set alight.
// `water`: falling water (CCubeMaterial::PortIsFallingWater), which shots ripple and splash
// instead of setting alight.
void NoteSurface(const CCubeSurface& surface, const CCubeModel& model, bool water);
// Updates the pushers, fires and rain splashes and sends this pass's foliage state, in its view
// space. Call once the pass's view matrix is set.
void Upload(const CStateManager& mgr);
// Draws the fires, smoke and rain splashes, after the world's transparent draws.
void DrawEffects(const CStateManager& mgr);
// Adds a flickering point light per burning plant (for the game's dynamic light list).
void AppendFireLights(rstl::vector< CLight >& lights);
// Forgets every plant and fire (a world change).
void Reset();
} // namespace PortFoliage
