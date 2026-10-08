#pragma once

#include "graphics_internal.hpp"
#include <string>
#include <vector>

namespace application {

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

} // namespace application