#include "application.hpp"

#include <imgui.h>

namespace application {

bool initialize() {
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);
}

void update([[maybe_unused]] double time) {
	ImGui::Begin("Hello world!");
	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	(void)fd;
}

} // namespace application