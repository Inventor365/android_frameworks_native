/*
 * Copyright 2021 The Android Open Source Project
 * Copyright 2026 Neoteric OS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define ATRACE_TAG ATRACE_TAG_GRAPHICS
#include "BlurFilter.h"
#include <SkBlendMode.h>
#include <SkCanvas.h>
#include <SkM44.h>
#include <SkPaint.h>
#include <SkRRect.h>
#include <SkRuntimeEffect.h>
#include <SkSize.h>
#include <SkString.h>
#include <SkSurface.h>
#include <SkTileMode.h>
#include <common/ThreadStateCrashLogger.h>
#include <common/trace.h>
#include <log/log.h>
#include <sys/system_properties.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>

#include "RuntimeEffectManager.h"

namespace android {
namespace renderengine {
namespace skia {

const SkString kEffectSource_BlurFilter_MixEffect(R"(
    uniform shader blurredInput;
    uniform shader originalInput;
    uniform float mixFactor;

    half4 main(float2 xy) {
        return half4(mix(originalInput.eval(xy), blurredInput.eval(xy), mixFactor)).rgb1;
    }
)");

// Glass slab rim: flat across the middle, rolling off along a squircle h(x) = (1 - (1 - x)^4)^(1/4),
// x being the depth into the rim. Snell's law bends the view outwards along it.
// lens: x = rim depth, y = shift at the very edge, z = refractive index, w = max tan(i - r).
// lensHeight: rim height / rim depth.
// lensRim: x = width, y = strength of the specular highlight along the rim.
#define LENS_SKSL                                                                   \
    "uniform float4 lens;\n"                                                       \
    "uniform float lensHeight;\n"                                                  \
    "uniform float2 lensRim;\n"                                                    \
    "float2 lensNormal(float2 rel, float2 h, float r) {\n"                         \
    "    float2 q = abs(rel) - (h - r);\n"                                         \
    "    float2 n = (q.x > 0.0 && q.y > 0.0) ? normalize(q)\n"                     \
    "             : (q.x > q.y ? float2(1.0, 0.0) : float2(0.0, 1.0));\n"          \
    "    return n * float2(rel.x < 0.0 ? -1.0 : 1.0, rel.y < 0.0 ? -1.0 : 1.0);\n" \
    "}\n"                                                                          \
    "float lensShift(float depth) {\n"                                             \
    "    float k = 1.0 - clamp(depth / lens.x, 0.0, 1.0);\n"                       \
    "    if (k <= 0.0 || lens.y <= 0.0) return 0.0;\n"                             \
    "    float k3 = k * k * k;\n"                                                  \
    "    float slope = lensHeight * k3 / pow(max(1.0 - k3 * k, 1e-4), 0.75);\n"    \
    "    float incident = atan(slope);\n"                                          \
    "    float refracted = asin(sin(incident) / lens.z);\n"                        \
    "    return lens.y * tan(incident - refracted) / lens.w;\n"                    \
    "}\n"

// Unified PLEnhanced + Snell Liquid Glass effect:
// Rounded slabs bend the backdrop via Snell's law with RGB chromatic dispersion along the rim,
// blend clarity near the curved edge, apply Rec.709 vibrancy/tone lift, and paint directional
// top-left specular rim + soft bottom-right inner shade matching PLEnhanced's GlassShader.LIQUID.
const SkString kEffectSource_LiquidGlassEffect(LENS_SKSL R"(
    uniform shader blurredInput;
    uniform shader originalInput;
    uniform float clarity;
    uniform float4 bounds;
    uniform float radius;
    uniform float band;
    uniform float dispersion;
    uniform float rim;
    uniform float densityPx;
    uniform float saturation;
    uniform float gain;
    uniform float lift;
    uniform float edgeClear;

    float roundRect(float2 p, float2 c, float2 h, float r) {
        float2 q = abs(p - c) - (h - r);
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    half4 main(float2 xy) {
        float2 c = (bounds.xy + bounds.zw) * 0.5;
        float2 h = (bounds.zw - bounds.xy) * 0.5;
        float r = min(radius, min(h.x, h.y));
        float d = roundRect(xy, c, h, r);
        float2 n = lensNormal(xy - c, h, r);

        float t = clamp(1.0 + d / max(band, 1.0), 0.0, 1.0);
        float shift = lensShift(-d);
        float2 offset = n * shift;
        float2 src = xy + offset;

        half3 rgb;
        if (dispersion > 0.001 && shift > 0.01) {
            rgb.r = blurredInput.eval(xy + offset * (1.0 + dispersion)).r;
            rgb.g = blurredInput.eval(src).g;
            rgb.b = blurredInput.eval(xy + offset * (1.0 - dispersion)).b;
        } else {
            rgb = blurredInput.eval(src).rgb;
        }

        float clear = clarity + (1.0 - clarity) * edgeClear * t * t;
        if (clear > 0.002) {
            half3 sharp;
            if (dispersion > 0.001 && shift > 0.01) {
                sharp.r = originalInput.eval(xy + offset * (1.0 + dispersion)).r;
                sharp.g = originalInput.eval(src).g;
                sharp.b = originalInput.eval(xy + offset * (1.0 - dispersion)).b;
            } else {
                sharp = originalInput.eval(src).rgb;
            }
            rgb = mix(rgb, sharp, half(clear));
        }

        half l = dot(rgb, half3(0.2126, 0.7152, 0.0722));
        rgb = mix(half3(l), rgb, half(saturation));
        rgb = rgb * half(gain) + half(lift);

        if (rim > 0.001) {
            float rimBand = smoothstep(-2.5 * densityPx, 0.0, d);
            float light = clamp(dot(n, normalize(float2(-0.5, -0.85))), 0.0, 1.0);
            float shade = clamp(dot(n, normalize(float2(0.5, 0.85))), 0.0, 1.0);
            float bend = t * t * t;
            rgb += half3(rim * rimBand * (0.14 + 0.42 * light));
            rgb += half3(rim * bend * 0.10 * light);
            rgb -= half3(rim * bend * 0.05 * shade);
        }
        return half4(clamp(rgb, 0.0, 1.0), 1.0);
    }
)");

// ColorOS Frosted Glass effect: multi-layer Luminosity + ColorDodge / Overlay frosted stack
// with perimeter directional rimLight and noise-dithered innerGlow.
const SkString kEffectSource_CosGlassEffect(LENS_SKSL R"(
    uniform shader blurredInput;
    uniform half4 modes;
    uniform half4 color0;
    uniform half4 color1;
    uniform half4 color2;
    uniform half4 color3;
    uniform half4 lift;
    uniform float4 bounds;
    uniform float radius;
    uniform float edge[16];
    uniform float glow[9];

    float rbox(float2 p, float2 b, float r) {
        float2 q = abs(p) - b + r;
        return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
    }
    float getDis(float2 pos, float2 halfSize) {
        float2 rect = abs(pos - halfSize);
        return rect.x + rect.y;
    }
    float random(float2 st) {
        return fract(sin(dot(st, float2(12.9898, 78.233))) * 43758.5453);
    }
    float softLight(float s, float d) {
        return s <= 0.5 ? d - (1.0 - 2.0 * s) * d * (1.0 - d) : d + (2.0 * s - 1.0) * (sqrt(d) - d);
    }
    half3 innerGlow(float2 xy, float2 pos, float2 halfSize, float corner, float d, half3 color) {
        float gray = clamp((dot(float3(0.2126, 0.7152, 0.0722), color) - 0.6) * 1.6, 0.0, 1.0);
        float2 fade = float2(glow[0], glow[1]) * (1.0 - gray) * (1.0 - gray);
        if (fade.x < 1e-4 && fade.y < 1e-4) {
            return color;
        }
        float fadeLine = -glow[2];
        float2 halfFade = halfSize * float2(glow[5], glow[6]);
        float fadeCorner = min(min(halfFade.x, halfFade.y), max(glow[2] * 1.2, corner));
        float fadeSdf = rbox(pos - float2(glow[3], glow[4]), halfFade, fadeCorner);
        float alpha = smoothstep(fadeLine, 0.0, fadeSdf) * smoothstep(fadeLine, 0.0, d);
        if (glow[8] > 1e-4) {
            alpha = pow(alpha, glow[7]) * glow[8] + (1.0 - glow[8]) * alpha;
        }
        alpha = alpha * (fade.x - fade.y) + fade.y;
        float n = clamp((random(xy * 0.013) - 0.5) * 0.008 + 0.5, 0.0, 1.0);
        alpha = clamp(softLight(n, alpha), 0.0, 1.0);
        return saturate(mix(color, color + 1.0, alpha));
    }
    half3 rimLight(float2 pos, float2 halfSize, float d, half3 color, half3 raw) {
        float2 lineNear = float2(edge[3], edge[4]);
        float2 lineFar = float2(edge[5], edge[6]);
        if (d < -max(lineNear.y, lineFar.y)) {
            return color;
        }
        float distanceAll = (halfSize.x + halfSize.y) * 2.0;
        float ratio = mod(edge[13] + halfSize.x / distanceAll * 0.5 + 0.5, 1.0);
        float2 dirAbs = abs(normalize(pos));
        float2 posN = min(float2(dirAbs.xy * halfSize.yx / max(float2(0.001), dirAbs.yx)), halfSize)
                * sign(pos);
        float flag = mix(-1.0, 1.0, step(ratio, 0.5)) * dot(float2(-halfSize.y, halfSize.x), posN);
        float disPoint = distanceAll * min(ratio, 1.0 - ratio) * 2.0;
        float disPos = getDis(posN, halfSize);
        float near = mix(disPoint + disPos, abs(disPoint - disPos), step(0.0, flag));
        near = min(2.0 * distanceAll - near, near);
        float nearAlpha = edge[7] * smoothstep(edge[10] * distanceAll, edge[9] * distanceAll, near);
        float farAlpha =
                edge[8] * smoothstep(edge[12] * distanceAll, edge[11] * distanceAll, distanceAll - near);
        float nearSdf = smoothstep(-lineNear.y, -lineNear.x, d);
        float farSdf = smoothstep(-lineFar.y, -lineFar.x, d);
        if (edge[15] > 1e-4) {
            nearSdf = pow(nearSdf, edge[14]) * edge[15] + (1.0 - edge[15]) * nearSdf;
            farSdf = pow(farSdf, edge[14]) * edge[15] + (1.0 - edge[15]) * farSdf;
        }
        float lineAlpha = clamp(nearAlpha * nearSdf + farAlpha * farSdf, 0.0, 1.0);
        float gray = clamp(dot(float3(0.2126, 0.7152, 0.0722), raw) * 1.2, 0.0, 1.0);
        float improve = (1.0 - gray) * (1.0 - gray);
        improve = improve * improve * 0.1;
        half3 lit = saturate(raw + improve);
        half3 lineColor = half3(edge[0], edge[1], edge[2]);
        lit = mix(2.0 * lit * lineColor, 1.0 - 2.0 * (1.0 - lit) * (1.0 - lineColor),
                  step(lineColor, half3(0.5)));
        return mix(color, saturate(lit), lineAlpha);
    }

    half overlay(half s, half d) {
        return (d < 0.5) ? 2.0 * s * d : 1.0 - 2.0 * (1.0 - s) * (1.0 - d);
    }
    half3 overlay(half3 s, half3 d) {
        return half3(overlay(s.x, d.x), overlay(s.y, d.y), overlay(s.z, d.z));
    }
    half colorDodge(half s, half d) { return d / max(half(1.0) - s, half(0.004)); }
    half3 colorDodge(half3 s, half3 d) {
        return half3(colorDodge(s.x, d.x), colorDodge(s.y, d.y), colorDodge(s.z, d.z));
    }
    half3 luminosity(half3 s, half3 d) {
        half dLum = dot(d, half3(0.3, 0.59, 0.11));
        half sLum = dot(s, half3(0.3, 0.59, 0.11));
        half3 c = d + (sLum - dLum);
        half minC = min(min(c.x, c.y), c.z);
        half maxC = max(max(c.x, c.y), c.z);
        if (minC < 0.0) {
            return sLum + ((c - sLum) * sLum) / (sLum - minC);
        } else if (maxC > 1.0) {
            return sLum + ((c - sLum) * (1.0 - sLum)) / (maxC - sLum);
        }
        return c;
    }
    half3 layer(half3 col, half mode, half4 c) {
        if (mode == 1) {
            col = mix(col, c.rgb, c.a);
        } else if (mode == 2) {
            col = mix(col, overlay(c.rgb, col), c.a);
        } else if (mode == 3) {
            col = mix(col, colorDodge(c.rgb, col), c.a);
        } else if (mode == 4) {
            col = mix(col, luminosity(c.rgb, col), c.a);
            col = mix(col, overlay(c.rgb, col), c.a);
        } else if (mode == 5) {
            col = mix(col, luminosity(c.rgb, col), c.a);
            col = mix(col, colorDodge(c.rgb, col), c.a);
        } else if (mode == 6) {
            half luminance = dot(col, half3(0.2126, 0.7152, 0.0722));
            col *= mix(half3(1), c.rgb, c.a * smoothstep(0.2, 1.0, luminance));
        }
        return col;
    }
    half4 main(float2 xy) {
        float2 halfSize = (bounds.zw - bounds.xy) * 0.5;
        float2 pos = xy - (bounds.xy + halfSize);
        float corner = min(radius, min(halfSize.x, halfSize.y));
        float d = rbox(pos, halfSize, corner);
        float2 src = xy;
        float2 n = float2(0.0);
        if (lens.y > 0.0 || lensRim.y > 0.0) {
            n = lensNormal(pos, halfSize, corner);
            src += n * lensShift(-d);
        }
        half4 col = blurredInput.eval(src);
        half3 rgb = layer(col.rgb, modes.x, color0);
        rgb = layer(rgb, modes.y, color1);
        rgb = layer(rgb, modes.z, color2);
        rgb = layer(rgb, modes.w, color3);
        rgb = rgb * lift.x + lift.y;
        half luma = dot(rgb, half3(0.2126, 0.7152, 0.0722));
        rgb = saturate(mix(half3(luma), rgb, lift.z));
        if (edge[7] + edge[8] > 0.0 || glow[0] > 0.0) {
            rgb = innerGlow(xy, pos, halfSize, corner, d, rgb);
            rgb = rimLight(pos, halfSize, d, rgb, col.rgb);
        }
        if (lensRim.y > 0.0) {
            float e = 1.0 - smoothstep(0.0, lensRim.x, -d);
            float facing = dot(n, float2(-0.5145, -0.8575));
            float spec = pow(max(facing, 0.0), 2.0) + 0.45 * pow(max(-facing, 0.0), 2.0);
            rgb = saturate(rgb + half(lensRim.y * e * (0.15 + 0.85 * spec)));
        }
        return half4(rgb, 1.0);
    }
)");

// Material IDs shared with com.android.internal.util.neoteric.YukiGlassConfig
static constexpr int32_t kMaterialPlainOptOut = -1;
static constexpr int32_t kMaterialAuto = 0;
static constexpr int32_t kMaterialCosDarkPanel = 5;
static constexpr int32_t kMaterialCosLightPanel = 6;
static constexpr int32_t kMaterialDockNeutral = 11;
static constexpr int32_t kMaterialLiquidPanel = 12;
static constexpr int32_t kMaterialLiquidSlabDark = 13;
static constexpr int32_t kMaterialLiquidSlabLight = 14;
static constexpr int32_t kMaterialPlainBlur = 15;

// Global glass modes (persist.sys.yuki.glass.style)
static constexpr int32_t kModeOff = 0;
static constexpr int32_t kModeBlur = 1;
static constexpr int32_t kModeFrosted = 2;
static constexpr int32_t kModeLiquid = 3;

// Lock-free cached system property reader: only re-parses when the property serial advances.
class CachedIntProperty {
public:
    CachedIntProperty(const char* name, int32_t fallback) : mName(name), mValue(fallback) {}

    int32_t get() {
        if (mInfo == nullptr) {
            const uint32_t area = __system_property_area_serial();
            if (area == mAreaSerial) return mValue;
            mAreaSerial = area;
            mInfo = __system_property_find(mName);
            if (mInfo == nullptr) return mValue;
        }
        const uint32_t serial = __system_property_serial(mInfo);
        if (serial != mSerial) {
            mSerial = serial;
            __system_property_read_callback(
                    mInfo,
                    [](void* cookie, const char*, const char* value, uint32_t) {
                        if (value != nullptr && value[0] != '\0') {
                            char* end = nullptr;
                            const long parsed = std::strtol(value, &end, 10);
                            if (end != value) {
                                *static_cast<int32_t*>(cookie) = static_cast<int32_t>(parsed);
                            }
                        }
                    },
                    &mValue);
        }
        return mValue;
    }

private:
    const char* mName;
    const prop_info* mInfo = nullptr;
    uint32_t mAreaSerial = UINT32_MAX;
    uint32_t mSerial = UINT32_MAX;
    int32_t mValue;
};

static bool propGlassEnabled() {
    static CachedIntProperty prop("persist.sys.yuki.glass.enabled", 1);
    return prop.get() != 0;
}

static int32_t propGlobalStyle() {
    static CachedIntProperty prop("persist.sys.yuki.glass.style", kModeLiquid);
    return std::clamp(prop.get(), kModeOff, kModeLiquid);
}

static int32_t propShadeMode() {
    static CachedIntProperty prop("persist.sys.yuki.glass.shade_mode", -1);
    const int32_t m = prop.get();
    return m >= kModeOff && m <= kModeLiquid ? m : propGlobalStyle();
}

static float propRefractionScale() {
    static CachedIntProperty prop("persist.sys.yuki.glass.refraction", 60);
    return std::clamp(prop.get(), 0, 100) * 0.015f; // 0..1.5 (default 0.90)
}

static float propDispersion() {
    static CachedIntProperty prop("persist.sys.yuki.glass.dispersion", 60);
    // Matches PLEnhanced GlassShader.DISPERSION (0.06 at 100%) scaled smoothly up to 0.08
    return std::clamp(prop.get(), 0, 100) * 0.00075f;
}

static float propDepthScale() {
    static CachedIntProperty prop("persist.sys.yuki.glass.depth", 100);
    return std::clamp(prop.get(), 20, 150) * 0.01f;
}

static float propVibrancyScale() {
    static CachedIntProperty prop("persist.sys.yuki.glass.vibrancy", 15);
    return std::max(0.0f, 1.0f + std::clamp(prop.get(), -50, 100) * 0.01f);
}

static float propRimStrength() {
    static CachedIntProperty prop("persist.sys.yuki.glass.rim", 80);
    return std::clamp(prop.get(), 0, 100) * 0.01f;
}

static float propFrostScale() {
    static CachedIntProperty prop("persist.sys.yuki.glass.frost", 50);
    return std::clamp(prop.get(), 0, 100) * 0.01f;
}

static float densityScale() {
    static const float scale = [] {
        char value[PROP_VALUE_MAX] = {};
        const int density =
                __system_property_get("ro.sf.lcd_density", value) > 0 ? std::atoi(value) : 0;
        return density > 0 ? density / 160.0f : 3.0f;
    }();
    return scale;
}

void BlurFilter::setDisplay(const std::string& /* namePlusId */) {
    mGlassEnabled = (mLiquidGlassEffect != nullptr || mCosGlassEffect != nullptr) &&
            propGlassEnabled();
    mGlobalStyle = mGlassEnabled ? propGlobalStyle() : kModeOff;
}

int32_t BlurFilter::resolveWindowMaterial(const std::string& layerName) const {
    if (!mGlassEnabled) {
        return kMaterialPlainOptOut;
    }
    if (layerName.starts_with("VRI-NotificationShade#")) {
        const int32_t mode = propShadeMode();
        if (mode == kModeLiquid) return kMaterialLiquidPanel;
        if (mode == kModeFrosted) return kMaterialCosDarkPanel;
        return kMaterialPlainOptOut;
    }
    return kMaterialPlainOptOut;
}

struct GlassLens {
    float depth = 1.0f;
    float shift = 0.0f;
    float ior = 1.5f;
    float scale = 1.0f;
    float height = 1.0f;
    float rimWidth = 0.0f;
    float rimStrength = 0.0f;
};

static bool shouldRefract(const SkRRect& region, int32_t material, bool glassEnabled,
                          int32_t globalStyle) {
    if (region.isRect()) return false;
    if (material == kMaterialLiquidSlabDark || material == kMaterialLiquidSlabLight) {
        return true;
    }
    if (!glassEnabled) return false;
    if (material == kMaterialDockNeutral && globalStyle == kModeLiquid) return true;
    if (material == kMaterialAuto && globalStyle == kModeLiquid) return true;
    return false;
}

static GlassLens computeGlassLens(const SkRRect& region, int32_t material, bool glassEnabled,
                                  int32_t globalStyle) {
    GlassLens lens;
    if (!shouldRefract(region, material, glassEnabled, globalStyle)) return lens;
    const SkRect& rect = region.rect();
    const float corner = region.radii(SkRRect::kUpperLeft_Corner).fX;
    const float half = std::min(rect.width(), rect.height()) * 0.5f;
    if (half < 1.0f) return lens;

    const float depthFactor = propDepthScale();
    const float baseCorner = corner > 1.0f ? corner : std::min(16.0f * densityScale(), half);
    lens.depth = std::clamp(baseCorner * depthFactor, std::min(6.0f, half), half);
    lens.shift = lens.depth * propRefractionScale();
    lens.ior = 1.5f;
    lens.height = 1.0f;
    lens.scale = std::tan(static_cast<float>(M_PI_2) - std::asin(1.0f / lens.ior));
    if (material == kMaterialDockNeutral) {
        lens.rimWidth = 1.5f * densityScale();
        lens.rimStrength = 0.35f * propRimStrength();
    }
    return lens;
}

static void setLensUniforms(SkRuntimeShaderBuilder& builder, const GlassLens& lens) {
    builder.uniform("lens") = SkV4{lens.depth, lens.shift, lens.ior, lens.scale};
    builder.uniform("lensHeight") = lens.height;
    builder.uniform("lensRim") = SkV2{std::max(lens.rimWidth, 0.5f), lens.rimStrength};
}

float BlurFilter::lensReach(const SkRRect& region, int32_t material) const {
    const float shift = computeGlassLens(region, material, mGlassEnabled, mGlobalStyle).shift;
    if (shift <= 0.0f) return 0.0f;
    return shift * (1.0f + propDispersion());
}

struct CosGlassLayers {
    SkV4 modes;
    SkV4 colors[4];
};

static SkV4 argb(uint32_t c) {
    return SkV4{((c >> 16) & 0xff) / 255.0f, ((c >> 8) & 0xff) / 255.0f, (c & 0xff) / 255.0f,
                (c >> 24) / 255.0f};
}

static const CosGlassLayers& cosGlassLayers(int32_t material) {
    static const CosGlassLayers inactive{SkV4{5, 2, 5, 3},
                                         {argb(0xA6000000), argb(0x33999999), argb(0x40404040),
                                          argb(0x667B7B7B)}};
    static const CosGlassLayers active{SkV4{5, 2, 5, 3},
                                       {argb(0xA6000000), argb(0x33999999), argb(0x7DE6E6E6),
                                        argb(0x40CCCCCC)}};
    static const CosGlassLayers lightInactive{SkV4{5, 2, 5, 3},
                                              {argb(0x92000000), argb(0x66999999),
                                               argb(0x19404040), argb(0x4D737373)}};
    static const CosGlassLayers lightActive{SkV4{5, 2, 5, 3},
                                            {argb(0x92000000), argb(0x66999999), argb(0x73E6E6E6),
                                             argb(0x40CCCCCC)}};
    static const CosGlassLayers panel{SkV4{5, 2, 0, 0},
                                      {argb(0xA6000000), argb(0x33999999), argb(0), argb(0)}};
    static const CosGlassLayers lightPanel{SkV4{5, 2, 0, 0},
                                           {argb(0x92000000), argb(0x66999999), argb(0), argb(0)}};
    static const CosGlassLayers card{SkV4{5, 3, 0, 0},
                                     {argb(0x59404040), argb(0x66737373), argb(0), argb(0)}};
    static const CosGlassLayers lightCard{SkV4{5, 3, 0, 0},
                                          {argb(0x19404040), argb(0x4D737373), argb(0), argb(0)}};
    static const CosGlassLayers headsUp{SkV4{5, 3, 0, 0},
                                        {argb(0x8C595959), argb(0xFF333333), argb(0), argb(0)}};
    static const CosGlassLayers lightHeadsUp{SkV4{5, 3, 0, 0},
                                             {argb(0x80BFBFBF), argb(0xCC333333), argb(0),
                                              argb(0)}};
    static const CosGlassLayers dock{SkV4{6, 0, 0, 0},
                                     {argb(0xFFD4D4D4), argb(0), argb(0), argb(0)}};
    switch (material) {
        case kMaterialDockNeutral:
            return dock;
        case 2:
            return active;
        case 7:
            return card;
        case 8:
            return lightCard;
        case 9:
            return headsUp;
        case 10:
            return lightHeadsUp;
        case kMaterialCosDarkPanel:
            return panel;
        case kMaterialCosLightPanel:
            return lightPanel;
        case 3:
            return lightInactive;
        case 4:
            return lightActive;
        default:
            return inactive;
    }
}

struct CosRimTemplate {
    float nearWidth;
    float farWidth;
    float alphaNear;
    float alphaFar;
    float fade[4];
    float glowAlpha;
    float glowOffset;
};

static const CosRimTemplate& cosRimTemplate(int32_t material) {
    static const CosRimTemplate inactiveDark{7, 4, 0.35f, 0.4f, {0.25f, 0.6f, 0.15f, 0.5f}, 0.07f,
                                             25};
    static const CosRimTemplate inactiveLight{7, 4, 0.25f, 0.6f, {0.25f, 0.6f, 0.15f, 0.5f},
                                              0.15f, 36};
    static const CosRimTemplate active{8, 2, 0.5f, 0.5f, {0.3f, 0.6f, 0.3f, 0.6f}, 0.15f, 36};
    static const CosRimTemplate headsUp{7, 4, 0.4f, 0.4f, {0.2f, 0.5f, 0.2f, 0.5f}, 0.07f, 25};
    static const CosRimTemplate lightHeadsUp{7, 4, 0.8f, 0.8f, {0.2f, 0.5f, 0.2f, 0.5f}, 0.15f,
                                             36};
    switch (material) {
        case 2:
        case 4:
            return active;
        case 9:
            return headsUp;
        case 10:
            return lightHeadsUp;
        case 3:
        case 8:
            return inactiveLight;
        default:
            return inactiveDark;
    }
}

static float cosRimRatio(float w, float h, float radius) {
    if (w <= 0 || h <= 0) return 0.5f;
    const float shortSide = std::min(w, h);
    const float corner = std::min(radius, shortSide / 2);
    const float side = (w * h - shortSide * shortSide) / (2 * shortSide);
    float pos = w >= h ? side + h / 2 : h / 2 - side;
    const float longSide = std::max(w, h);
    const float stretch = radius >= shortSide / 2
            ? std::clamp((longSide - shortSide) / (shortSide * 0.25f), 0.0f, 1.0f)
            : 1.0f;
    pos -= corner / 2 * stretch;
    return pos * 0.5f / (w + h) + 0.5f;
}

static SkMatrix getShaderTransform(const SkCanvas* canvas, const SkRect& blurRect,
                                   const float scale, const float zoomScale) {
    auto matrix = SkMatrix::Scale(scale, scale);
    matrix.postConcat(SkMatrix::Translate(blurRect.fLeft, blurRect.fTop));
    if (zoomScale != 1.0f) {
        matrix.postScale(zoomScale, zoomScale);
        matrix.postTranslate(blurRect.width() * (1 - zoomScale) / 2.0f,
                             blurRect.height() * (1 - zoomScale) / 2.0f);
    }
    SkMatrix drawInverse;
    if (canvas != nullptr && canvas->getTotalMatrix().invert(&drawInverse)) {
        matrix.postConcat(drawInverse);
    }
    return matrix;
}

BlurFilter::BlurFilter(RuntimeEffectManager& effectManager, const float maxCrossFadeRadius)
      : mMaxCrossFadeRadius(maxCrossFadeRadius),
        mMixEffect(effectManager.mKnownEffects[kBlurFilter_MixEffect]),
        mLiquidGlassEffect(effectManager.mKnownEffects[kLiquidGlassEffect]),
        mCosGlassEffect(effectManager.mKnownEffects[kCosGlassEffect]) {}

float BlurFilter::getMaxCrossFadeRadius() const {
    return mMaxCrossFadeRadius;
}

void BlurFilter::drawBlurRegion(SkCanvas* canvas, const SkRRect& effectRegion,
                                const uint32_t blurRadius, const float zoomScale,
                                const float blurAlpha, const SkRect& blurRect,
                                sk_sp<SkImage> blurredImage, sk_sp<SkImage> input,
                                int32_t material) {
    SFTRACE_CALL();

    SkPaint paint;
    paint.setAlphaf(blurAlpha);

    auto blurMatrix = getShaderTransform(canvas, blurRect, kInverseInputScale, zoomScale);
    SkSamplingOptions linearSampling(SkFilterMode::kLinear, SkMipmapMode::kNone);
    const auto blurShader = blurredImage->makeShader(SkTileMode::kMirror, SkTileMode::kMirror,
                                                     linearSampling, &blurMatrix);

    // Resolve material == 0 (Auto) against global Yuki's Addons style.
    if (material == kMaterialAuto && mGlassEnabled) {
        if (mGlobalStyle == kModeFrosted) {
            material = effectRegion.isRect() ? kMaterialCosDarkPanel : 1;
        } else if (mGlobalStyle == kModeLiquid) {
            material = effectRegion.isRect() ? kMaterialLiquidPanel : kMaterialLiquidSlabDark;
        } else {
            material = kMaterialPlainBlur;
        }
    }

    // 1. ColorOS Frosted Glass (materials 1..11)
    if (material >= 1 && material <= kMaterialDockNeutral && mCosGlassEffect != nullptr) {
        SkRuntimeShaderBuilder glass(mCosGlassEffect);
        glass.child("blurredInput") = blurShader;
        const CosGlassLayers& layers = cosGlassLayers(material);
        // Scale frosted layer alpha smoothly by the user's Frost slider (default 50 -> 1.0x).
        const float frostFactor = material == kMaterialDockNeutral
                ? 1.0f
                : std::clamp(0.45f + propFrostScale() * 1.1f, 0.35f, 1.55f);
        auto scaledColor = [frostFactor](SkV4 c) -> SkV4 {
            c.w = std::clamp(c.w * frostFactor, 0.0f, 1.0f);
            return c;
        };
        glass.uniform("modes") = layers.modes;
        glass.uniform("color0") = scaledColor(layers.colors[0]);
        glass.uniform("color1") = scaledColor(layers.colors[1]);
        glass.uniform("color2") = scaledColor(layers.colors[2]);
        glass.uniform("color3") = scaledColor(layers.colors[3]);

        const float sat = propVibrancyScale();
        glass.uniform("lift") = material == kMaterialDockNeutral
                ? SkV4{1.0f, 0.0f, sat, 0.0f}
                : SkV4{1.0f, 0.0f, sat, 0.0f};

        const SkRect& rect = effectRegion.rect();
        const float corner = effectRegion.radii(SkRRect::kUpperLeft_Corner).fX;
        float edge[16] = {};
        float glow[9] = {};
        if (!effectRegion.isRect() &&
            ((material >= 1 && material <= 4) || (material >= 7 && material <= 10))) {
            const CosRimTemplate& rim = cosRimTemplate(material);
            const float px = 0.875f;
            const float rimStrength = propRimStrength();
            const float edgeValues[16] = {1, 1, 1,
                                          rim.nearWidth * px, rim.nearWidth * px * 2,
                                          rim.farWidth * px, rim.farWidth * px * 2,
                                          rim.alphaNear * rimStrength, rim.alphaFar * rimStrength,
                                          rim.fade[0], rim.fade[1], rim.fade[2], rim.fade[3],
                                          cosRimRatio(rect.width(), rect.height(), corner), 10,
                                          0.5f};
            std::copy(std::begin(edgeValues), std::end(edgeValues), edge);
            const float glowValues[9] = {rim.glowAlpha * rimStrength, 0, 40 * px, 0,
                                         rim.glowOffset * px, 1.3f, 1.3f, 20, 0.5f};
            std::copy(std::begin(glowValues), std::end(glowValues), glow);
        }
        glass.uniform("bounds") = SkV4{rect.fLeft, rect.fTop, rect.fRight, rect.fBottom};
        glass.uniform("radius") = corner;
        glass.uniform("edge").set(edge, 16);
        glass.uniform("glow").set(glow, 9);
        setLensUniforms(glass, computeGlassLens(effectRegion, material, mGlassEnabled,
                                                mGlobalStyle));
        paint.setShader(glass.makeShader());
        paint.setAntiAlias(!effectRegion.isRect());
        if (effectRegion.isRect()) {
            canvas->drawRect(rect, paint);
        } else {
            canvas->drawRRect(effectRegion, paint);
        }
        return;
    }

    // 2. PLEnhanced / iOS Liquid Glass (materials 12, 13, 14)
    const bool isLiquidMaterial = material == kMaterialLiquidPanel ||
            material == kMaterialLiquidSlabDark || material == kMaterialLiquidSlabLight;
    if (isLiquidMaterial && mLiquidGlassEffect != nullptr &&
        (!effectRegion.isRect() || blurRadius >= mMaxCrossFadeRadius)) {
        const bool slab = !effectRegion.isRect() &&
                (material == kMaterialLiquidSlabDark || material == kMaterialLiquidSlabLight);
        const bool isLight = material == kMaterialLiquidSlabLight;
        const SkRect& rect = effectRegion.rect();
        const float radius = slab ? effectRegion.radii(SkRRect::kUpperLeft_Corner).fX : 0.0f;
        const float shortSide = std::min(rect.width(), rect.height());
        const float depthScale = propDepthScale();
        const float band = slab
                ? std::clamp(std::max(radius, 12.0f * densityScale()) * 0.65f * depthScale,
                             6.0f, std::max(6.0f, shortSide * 0.5f))
                : 1.0f;

        SkRuntimeShaderBuilder glass(mLiquidGlassEffect);
        glass.child("blurredInput") = blurShader;
        SkMatrix inputMatrix;
        if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
            ALOGE("matrix was unable to be inverted");
        }
        glass.child("originalInput") =
                input != nullptr
                        ? input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror,
                                            linearSampling, inputMatrix)
                        : blurShader;
        glass.uniform("clarity") = 0.0f;
        glass.uniform("bounds") = SkV4{rect.fLeft, rect.fTop, rect.fRight, rect.fBottom};
        glass.uniform("radius") = radius;
        glass.uniform("band") = band;
        glass.uniform("dispersion") = slab ? propDispersion() : 0.0f;
        glass.uniform("densityPx") = densityScale();
        setLensUniforms(glass, computeGlassLens(effectRegion, material, mGlassEnabled,
                                                mGlobalStyle));
        const float sat = propVibrancyScale();
        const float frost = propFrostScale();
        const float rimStrength = propRimStrength();
        if (slab) {
            glass.uniform("rim") = rimStrength * (isLight ? 0.72f : 0.85f);
            glass.uniform("saturation") = sat * 1.08f;
            glass.uniform("gain") = isLight ? (1.0f + 0.06f * frost) : (0.94f - 0.14f * frost);
            glass.uniform("lift") = isLight ? (0.04f + 0.06f * frost) : 0.025f;
            glass.uniform("edgeClear") = 0.14f;
        } else {
            // Whole-window liquid glass backdrop (Shade / Settings / Recents / Drawer):
            // rich, luminous colors without distorting display edges.
            glass.uniform("rim") = 0.0f;
            glass.uniform("saturation") = std::clamp(sat * 0.95f, 0.5f, 1.8f);
            glass.uniform("gain") = 1.03f;
            glass.uniform("lift") = 0.02f;
            glass.uniform("edgeClear") = 0.0f;
        }
        paint.setShader(glass.makeShader());
        if (slab) {
            paint.setAntiAlias(true);
            canvas->drawRRect(effectRegion, paint);
        } else {
            if (blurAlpha == 1.0f) {
                paint.setBlendMode(SkBlendMode::kSrc);
            }
            canvas->drawRect(rect, paint);
        }
        return;
    }

    // 3. Standard Blur / Plain Opt-out (Kawase2 / Gaussian with smooth crossfade)
    if (blurRadius < mMaxCrossFadeRadius && mMixEffect != nullptr && input != nullptr) {
        SkMatrix inputMatrix;
        if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
            ALOGE("matrix was unable to be inverted");
        }
        SkRuntimeShaderBuilder blurBuilder(mMixEffect);
        blurBuilder.child("blurredInput") = blurShader;
        blurBuilder.child("originalInput") =
                input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror, linearSampling,
                                  inputMatrix);
        blurBuilder.uniform("mixFactor") = blurRadius / mMaxCrossFadeRadius;
        paint.setShader(blurBuilder.makeShader());
    } else {
        paint.setShader(blurShader);
    }

    if (effectRegion.isRect()) {
        if (blurAlpha == 1.0f && material >= 0 && material != kMaterialPlainBlur) {
            paint.setBlendMode(SkBlendMode::kSrc);
        }
        canvas->drawRect(effectRegion.rect(), paint);
    } else {
        paint.setAntiAlias(true);
        canvas->drawRRect(effectRegion, paint);
    }
}

} // namespace skia
} // namespace renderengine
} // namespace android
