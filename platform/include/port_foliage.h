#pragma once

// Interactive foliage (port_debug.h InteractiveFoliage): plants bend away from Samus and from
// shots passing through them, and sway back after.
class CStateManager;

namespace PortFoliage {
// Whether a texture (TXTR asset id) is a plant's: leaves, ferns, grass, hanging vines and moss.
bool IsFoliageTexture(unsigned int assetId);
// Updates the pushers (Samus, live shots, and what let go recently) and sends them for the pass
// about to draw, in its view space. Call once the pass's view matrix is set.
void Upload(const CStateManager& mgr);
} // namespace PortFoliage
