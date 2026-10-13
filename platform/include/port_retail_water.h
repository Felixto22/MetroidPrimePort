#pragma once

#include <cstdint>

class CStateManager;
class CVector3f;
class CColor;

// Enhanced water for the retail game: a water or poison object with no Remastered surface
// (port_room_liquid.h) is drawn by aurora's water shader (aurora/water.hpp) instead of its
// fluid plane. The surface is a grid over the object's box at the height the game keeps it,
// with two gentle waves, a procedural ripple normal map, Fresnel reflections marched through
// the scene in screen space, the scene behind seen through the water and tinted by depth with
// the object's own underwater colour, soft edges where it meets the ground, and foam along
// them. The object's splashes, sounds, damage and physics are the game's. Setting
// `enhanced_water` (PortDebug::EnhancedWater); not in the X-ray or thermal visors.
namespace PortRetailWater {
// Draws the object's surface; false to leave it to the retail fluid plane. `boxMin`/`boxMax` are
// its world bounds, `surfaceZ` the height of its surface, `fluidType` its CFluidPlane type and
// `insideColor` its underwater fog colour.
bool Draw(const CStateManager& mgr, uint32_t uid, const CVector3f& boxMin, const CVector3f& boxMax, float surfaceZ,
          int fluidType, const CColor& insideColor);
// Forgets the cached meshes (a world change).
void Reset();
} // namespace PortRetailWater
