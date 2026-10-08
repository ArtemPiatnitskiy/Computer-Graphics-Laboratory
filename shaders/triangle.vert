#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragColor;

layout(std140, set = 0, binding = 0) uniform Transform {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 color;
} transform;

void main() {
    vec4 p = transform.model * vec4(inPosition, 1.0);

    mat3 normalMatrix = transpose(inverse(mat3(transform.model)));
    fragNormal = normalMatrix * inNormal;

    fragColor = inColor * transform.color.rgb;

    gl_Position = transform.projection * transform.view * p;
} 