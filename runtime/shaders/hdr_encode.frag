#version 450
// The HDR pass of the present chain (runtime/vk_hdr.c vk_hdr_record): the 8-bit sRGB picture after FSR and SMAA onto an
// HDR10 swapchain image. Linearise (sRGB EOTF), lift the highlights above a knee, put the picture's white at p.x nits
// and write PQ code values in Rec. 2020. p.x: nits of the picture's white, p.y: the display's peak in nits, p.z: the
// knee as a linear value. PQ takes absolute nits, so linear 1.0 times p.x is the same scaling as scRGB's "game nits / 80"
// followed by its 80-nit unit.
// The lift: with m the largest channel, m <= knee is left alone and above it m becomes m + (R - 1) * t^2, t =
// (m - knee) / (1 - knee), which meets the knee with slope 1 and takes 1.0 to R = peak / white (the display's headroom
// over the picture's white). All three channels are scaled by the same factor, so hue and saturation are kept.
// PQ: SMPTE ST 2084; the Rec. 709 to Rec. 2020 matrix: ITU-R BT.2087.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D src;
layout(push_constant) uniform P { vec4 p; } pc;

vec3 srgb_decode(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

vec3 pq(vec3 nits) {
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), vec3(m1));
    return pow((c1 + c2 * y) / (1.0 + c3 * y), vec3(m2));
}

void main() {
    vec3 c = srgb_decode(clamp(texture(src, uv).rgb, 0.0, 1.0));
    float knee = pc.p.z, headroom = max(pc.p.y / pc.p.x, 1.0);
    float m = max(c.r, max(c.g, c.b));
    if (m > knee) {
        float t = (m - knee) / (1.0 - knee);
        c *= (m + (headroom - 1.0) * t * t) / m;
    }
    const mat3 to2020 = mat3(0.6274, 0.0691, 0.0164, 0.3293, 0.9195, 0.0880, 0.0433, 0.0114, 0.8956);
    vec3 nits = min(max(to2020 * c, vec3(0.0)) * pc.p.x, vec3(pc.p.y));
    color = vec4(pq(nits), 1.0);
}
