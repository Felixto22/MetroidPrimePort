// Cinematic look for the retail picture (port_cinematic.h).
#include "port_cinematic.h"

#include "port_debug.h"

#include "Kyoto/Graphics/CGraphics.hpp"
#include "Kyoto/Math/CTransform4f.hpp"
#include "Kyoto/Math/CVector3f.hpp"

#include <dolphin/gx/GXExtra.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace PortCinematic {
namespace {
constexpr uint32_t kGradeId = 0x43494e45; // 'CINE'
constexpr int kLut = 33;
float sSun[3] = {};

float SrgbEnc(float x) { return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f; }
float SrgbDec(float x) { return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f); }

// The grade, on display values: an S-curve for contrast, a little more colour, shadows leaning
// teal and highlights warm.
void Grade(float rgb[3]) {
  float d[3];
  for (int c = 0; c < 3; ++c) {
    const float x = std::clamp(SrgbEnc(rgb[c]), 0.f, 1.f);
    const float s = x * x * (3.f - 2.f * x);
    d[c] = x + (s - x) * 0.4f;
  }
  const float luma = 0.2126f * d[0] + 0.7152f * d[1] + 0.0722f * d[2];
  for (float& v : d) {
    v = luma + (v - luma) * 1.15f;
  }
  const float shadow = std::clamp(1.f - luma * 2.f, 0.f, 1.f);
  const float high = std::clamp(luma * 2.f - 1.f, 0.f, 1.f);
  d[0] += 0.025f * high - 0.01f * shadow;
  d[1] += 0.006f * shadow + 0.008f * high;
  d[2] += 0.02f * shadow - 0.02f * high;
  for (int c = 0; c < 3; ++c) {
    rgb[c] = SrgbDec(std::clamp(d[c], 0.f, 1.f));
  }
}

void RegisterGrade() {
  static bool done = false;
  if (done) {
    return;
  }
  done = true;
  std::vector< uint8_t > lut(size_t(kLut) * kLut * kLut * 4);
  size_t i = 0;
  for (int b = 0; b < kLut; ++b) {
    for (int g = 0; g < kLut; ++g) {
      for (int r = 0; r < kLut; ++r) {
        float rgb[3] = {float(r) / (kLut - 1), float(g) / (kLut - 1), float(b) / (kLut - 1)};
        Grade(rgb);
        for (int c = 0; c < 3; ++c) {
          lut[i++] = uint8_t(std::clamp(rgb[c] * 255.f + 0.5f, 0.f, 255.f));
        }
        lut[i++] = 255;
      }
    }
  }
  GXPortColorGradeLut(kGradeId, lut.data());
}
} // namespace

bool Retail(float& threshold, float tints[5][3], float tone[3][4], uint32_t& grade) {
  if (!PortDebug::CinematicLook()) {
    return false;
  }
  RegisterGrade();
  // The retail EFB is display-referred and tops out at 1: read it through a curve that is the
  // identity up to 0.7 and a shoulder above (y = 0.7 + 0.3 s / (1 + s), s = 10/3 (x - 0.7)), so
  // that only the nearly white light - shots, the sky, glows - comes back bright enough to bloom.
  std::memset(tone, 0, sizeof(float) * 12);
  tone[1][0] = 1.f;  // the line's slope
  tone[1][1] = 0.f;  // and offset
  tone[1][2] = 0.f;  // no toe
  tone[1][3] = 0.7f; // where the shoulder starts
  tone[2][0] = 0.3f;
  tone[2][1] = 10.f / 3.f;
  tone[2][2] = -0.7f * 10.f / 3.f;
  tone[2][3] = 0.7f;
  threshold = 1.15f;
  // Coarsest level first; the last weights the bright pass.
  const float levels[5] = {0.55f, 0.45f, 0.35f, 0.3f, 1.f};
  const float warm[3] = {1.f, 0.93f, 0.85f};
  for (int l = 0; l < 5; ++l) {
    for (int c = 0; c < 3; ++c) {
      tints[l][c] = levels[l] * warm[c];
    }
  }
  grade = kGradeId;

  // Extras: the lens star always; light shafts when the room has a sun in front of the camera.
  float sunUv[2] = {0.f, 0.f};
  bool inFront = false;
  if (sSun[0] != 0.f || sSun[1] != 0.f || sSun[2] != 0.f) {
    const CTransform4f& view = CGraphics::GetViewMatrix();
    // Towards the sun, in the camera's frame: right, forward, up.
    const CVector3f toSun(-sSun[0], -sSun[1], -sSun[2]);
    const float vx = CVector3f::Dot(toSun, view.GetRight());
    const float vy = CVector3f::Dot(toSun, view.GetForward());
    const float vz = CVector3f::Dot(toSun, view.GetUp());
    const CGraphics::CProjectionState& proj = CGraphics::GetProjectionState();
    if (vy > 0.05f && proj.IsPerspective()) {
      const float n = proj.GetNear();
      const float x = n * vx / vy, y = n * vz / vy;
      const float ndcX = (2.f * x - (proj.GetRight() + proj.GetLeft())) / (proj.GetRight() - proj.GetLeft());
      const float ndcY = (2.f * y - (proj.GetTop() + proj.GetBottom())) / (proj.GetTop() - proj.GetBottom());
      sunUv[0] = ndcX * 0.5f + 0.5f;
      sunUv[1] = 0.5f - ndcY * 0.5f;
      inFront = true;
    }
  }
  GXPortPostProcessExtras(0.7f, inFront ? 0.6f : 0.f, sunUv, inFront ? GX_TRUE : GX_FALSE);
  return true;
}

void SetSun(const float dir[3]) {
  const float len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  for (int i = 0; i < 3; ++i) {
    sSun[i] = len > 1e-4f ? dir[i] / len : 0.f;
  }
}

GunShine::GunShine() : on(PortDebug::CinematicLook()) {
  if (on) {
    GXPortSetSpecular(0.85f);
  }
}

GunShine::~GunShine() {
  if (on) {
    GXPortSetSpecular(0.f);
  }
}
} // namespace PortCinematic
