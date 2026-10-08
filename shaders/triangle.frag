#version 450

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDirection = normalize(vec3(-0.5, 0.8, -1.0));
    float diffuse = max(dot(normal, lightDirection), 0.0);
    float brightness = 0.2 + 0.8 * diffuse;
    outColor = vec4(fragColor * brightness, 1.0);
}