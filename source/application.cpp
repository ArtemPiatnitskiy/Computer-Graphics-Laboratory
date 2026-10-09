#include "application.hpp"
#include "graphics_internal.hpp"
#include <exception>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <numbers>
#include <cmath>
#include <array>
#include <iostream>

#include <imgui.h>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace application {

	namespace {
		constexpr uint32_t OBJECTS_COUNT = 2;

		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;

		VkBuffer vertex_buffer = VK_NULL_HANDLE;
		VmaAllocation vertex_allocation = nullptr;

		VkBuffer index_buffer = VK_NULL_HANDLE;
		VmaAllocation index_allocation = nullptr;

		VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

		uint32_t index_count = 0;

		struct Vertex {
			float x, y, z, nx, ny, nz;
			float red, green, blue;
		};
		struct Mesh {
			std::vector<Vertex> vertices;
			std::vector<uint32_t> indices;
		};

		struct TransformUniforms {
			alignas(16) glm::mat4 model;
			alignas(16) glm::mat4 view;
			alignas(16) glm::mat4 projection;
			alignas(16) glm::vec4 color;
		};

		struct ObjectState {
			glm::vec3 position{0.0f};
			float rotation_x_degrees = 0.0f;
			glm::vec3 scale{1.0f};
			glm::vec3 color{1.0f};
		};

		std::array<ObjectState, OBJECTS_COUNT> object_states{};
		int selected_object = 0;

		struct ObjectResources {
			VkBuffer uniform_buffer = VK_NULL_HANDLE;
			VmaAllocation uniform_allocation = nullptr;
			VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
		};

		std::array<ObjectResources, OBJECTS_COUNT> objects{};

		bool is_animate = false;

		double animation_phase = 0.0; // Положение на траектории
		float animation_speed = 1.0f; // радиан фазы в секунду
		float trajectory_size = 0.6f;

		int projection_mod = 1; // 1 - перспективная; 2 - ортографическая

	Mesh makeTorus(float R, float r, uint32_t rings, uint32_t sides) {
		Mesh mesh;
		for (uint32_t ring = 0; ring < rings; ++ring) {
			float u = 2.0f * std::numbers::pi_v<float> * ring / rings;
			for (uint32_t side = 0; side < sides; ++side) {
				float v = 2.0f * std::numbers::pi_v<float> * side / sides;

				float x = (R + r * std::cos(v)) * std::cos(u);
				float y = (R + r * std::cos(v)) * std::sin(u);
				float z = r * std::sin(v);

				float nx = std::cos(v) * std::cos(u);
				float ny = std::cos(v) * std::sin(u);
				float nz = std::sin(v);

				const float red   = 0.5f * (x / (R + r) + 1.0f);
				const float green = 0.5f * (y / (R + r) + 1.0f);
				const float blue  = 0.5f * (z / r + 1.0f);

				mesh.vertices.push_back({
					x, y, z,
					nx, ny, nz,
					red, green, blue 
			});

			}
		}

		for (uint32_t ring = 0; ring < rings; ++ring) {
			for (uint32_t side = 0; side < sides; ++side) {
				uint32_t nextRing = (ring + 1) % rings;
				uint32_t nextSide = (side + 1) % sides;

				uint32_t a = ring     * sides + side;
				uint32_t b = nextRing * sides + side;
				uint32_t c = ring     * sides + nextSide;
				uint32_t d = nextRing * sides + nextSide;

				mesh.indices.insert(mesh.indices.end(), { a, c, b, c, d, b });
			}
		}

		return mesh;
	}

	std::vector<uint32_t> readShaderFile(const std::string& path) {
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open shader file: " + path);
		}
		const auto size = file.tellg();

		if (size <= 0 || size % sizeof(uint32_t) != 0) {
			throw std::runtime_error("Shader file size is not a multiple of 4 bytes: " + path);
		}
		
		std::vector<uint32_t> buffer(size / sizeof(uint32_t));

		file.seekg(0);
		if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
			throw std::runtime_error("Failed to read shader file: " + path);
		}
		file.close();

		return buffer;
	}

	VkShaderModule createShaderModule(const std::string& path) {
		std::vector<uint32_t> shader_code = readShaderFile(path);
		const VkShaderModuleCreateInfo create_info = {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = shader_code.size() * sizeof(uint32_t),
			.pCode = shader_code.data()
		};

		VkShaderModule shader_module = VK_NULL_HANDLE;
		if (vkCreateShaderModule(
			graphics::internal::context.device,
			&create_info,
			nullptr,
			&shader_module
		) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create shader module: " + path);
		}
		return shader_module;
	}

	void createMeshBuffers(const Mesh& mesh) {
		
		VkDeviceSize vertex_bytes = mesh.vertices.size() * sizeof(Vertex);
		VkDeviceSize index_bytes = mesh.indices.size() * sizeof(uint32_t);

		const VkBufferCreateInfo vertex_buffer_info = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = vertex_bytes,
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE
		};

		const VkBufferCreateInfo index_buffer_info = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = index_bytes,
			.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE
		};

		const VmaAllocationCreateInfo allocation_info = {
			.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | 
					VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO,
		};

		if (
			vmaCreateBuffer(
			graphics::internal::context.allocator,
			&vertex_buffer_info,
			&allocation_info,
			&vertex_buffer,
			&vertex_allocation,
			nullptr
		) != VK_SUCCESS || 
			vmaCreateBuffer(
			graphics::internal::context.allocator,
			&index_buffer_info,
			&allocation_info,
			&index_buffer,
			&index_allocation,
			nullptr
		) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create index buffer");
		}

		if (
			vmaCopyMemoryToAllocation(
			graphics::internal::context.allocator,
			mesh.vertices.data(),
			vertex_allocation,
			0,
			vertex_bytes
		) != VK_SUCCESS ||
			vmaCopyMemoryToAllocation(
			graphics::internal::context.allocator, 
			mesh.indices.data(),
			index_allocation,
			0,
			index_bytes
		) != VK_SUCCESS) {
			throw std::runtime_error("Failed to copy index data to buffer");
		}
	}

	void createUniformBuffer(ObjectResources& object) {
		VkDeviceSize uniform_bytes = sizeof(TransformUniforms);

		const VkBufferCreateInfo uniform_buffer_info = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = uniform_bytes,
			.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, // Используем как буфер для uniform данных
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE // доступ одной семьи очередей в каждый момент времени 
			// (То есть из другого семейства очередей сюда не придут)
		};

		const VmaAllocationCreateInfo allocation_info = {
			.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | // отображение памяти в адресное пространство CPU
					 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, // Доступ к памяти будет последовательным для записи
			.usage = VMA_MEMORY_USAGE_AUTO // Автоматический выбор типа памяти
		};

		if (vmaCreateBuffer(
			graphics::internal::context.allocator,
			&uniform_buffer_info,
			&allocation_info,
			&object.uniform_buffer,
			&object.uniform_allocation,
			nullptr
		) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create uniform buffer");
		}
	}

	void updateUniformBuffer(const TransformUniforms& transform, ObjectResources& object) {
		if (vmaCopyMemoryToAllocation(
			graphics::internal::context.allocator,
			&transform,
			object.uniform_allocation,
			0,
			sizeof(TransformUniforms)
		) != VK_SUCCESS) {
			throw std::runtime_error("Failed to update uniform buffer");
		}
	}

	void destroyUniformBuffer(ObjectResources& object) {
		vmaDestroyBuffer(
			graphics::internal::context.allocator,
			object.uniform_buffer,
			object.uniform_allocation
		);
	}

	VkDescriptorSetLayout createDescriptorSetLayout() {
		const VkDescriptorSetLayoutBinding uniform_binding = {
			.binding = 0, // шейдер будет обращаться к записи 0
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1, // в записи один дескриптор буфера
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, // доступ к буферу будет у вершинного шейдера
			.pImmutableSamplers = nullptr // не используем сэмплеры
		};

		const VkDescriptorSetLayoutCreateInfo layout_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 1,
			.pBindings = &uniform_binding
		};

		VkDescriptorSetLayout result_layout = VK_NULL_HANDLE;

		if (vkCreateDescriptorSetLayout(
			graphics::internal::context.device,
			&layout_info,
			nullptr,
			&result_layout
		) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor set layout");
		}
		return result_layout;
	}

	VkDescriptorPool createDescriptorPool() {
		const VkDescriptorPoolSize pool_size = {
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = OBJECTS_COUNT
		};

		const VkDescriptorPoolCreateInfo pool_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = OBJECTS_COUNT,
			.poolSizeCount = 1,
			.pPoolSizes = &pool_size,
		};

		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

		if (vkCreateDescriptorPool(
			graphics::internal::context.device,
			&pool_info,
			nullptr,
			&descriptor_pool
		) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor pool");
		}
		return descriptor_pool;
	}

	VkDescriptorSet allocateDescriptorSet() {
		const VkDescriptorSetAllocateInfo allocate_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &descriptor_set_layout
		};

		VkDescriptorSet result_set = VK_NULL_HANDLE;
		if (vkAllocateDescriptorSets(
			graphics::internal::context.device,
			&allocate_info,
			&result_set
		) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate descriptor set");
		}
		return result_set;
	}

	void writeUniformBufferDescriptor(ObjectResources& object) {
		const VkDescriptorBufferInfo buffer_info = {
			.buffer = object.uniform_buffer,
			.offset = 0,
			.range = sizeof(TransformUniforms)
		};

		const VkWriteDescriptorSet descriptor_write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = object.descriptor_set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info
		};

		vkUpdateDescriptorSets(
			graphics::internal::context.device,
			1, &descriptor_write,
			0, nullptr
		);
	}

	VkPipelineLayout createPipelineLayout() {
		
		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

		const VkPipelineLayoutCreateInfo pipeline_layout_info = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout
		};

		if (vkCreatePipelineLayout(
			graphics::internal::context.device,
			&pipeline_layout_info,
			nullptr,
			&pipeline_layout
		) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create pipeline layout");
		}
		return pipeline_layout;
	}

	VkVertexInputBindingDescription makeVertexBindingDescription() {
		const VkVertexInputBindingDescription binding = {
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		};
		return binding;
	}

	std::array<VkVertexInputAttributeDescription, 3> makeVertexPositionAttributeDescriptions() {
		return {
			VkVertexInputAttributeDescription{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, x)
			},
			VkVertexInputAttributeDescription{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, nx)
			},
			VkVertexInputAttributeDescription{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, red)
			}
		};
	}

	VkPipelineInputAssemblyStateCreateInfo makeInputAssemblyState() {
		const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE
		};
		return input_assembly;
	}

	VkPipelineViewportStateCreateInfo makeViewportState() {
		const VkPipelineViewportStateCreateInfo viewport_state = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1
		};
		return viewport_state;
	}

	VkPipelineRasterizationStateCreateInfo makeRasterizationState() {
		const VkPipelineRasterizationStateCreateInfo rasterizer = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f
		};
		return rasterizer;
	}

	VkPipelineMultisampleStateCreateInfo makeMultisampleState() {
		const VkPipelineMultisampleStateCreateInfo multisampling = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		};
		return multisampling;
	}

	VkPipelineDepthStencilStateCreateInfo makeDepthStencilState() {
		const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS,
		};
		return depth_stencil;
	}

	VkPipelineColorBlendAttachmentState makeColorBlendAttachmentState() {
		const VkPipelineColorBlendAttachmentState color_attachment = {
			.blendEnable = VK_FALSE,
			.colorWriteMask = 
							VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
							VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};
		return color_attachment;
	}

	VkPipeline createGraphicsPipeline(VkPipelineLayout pipeline_layout) {

		VkPipeline result_pipeline = VK_NULL_HANDLE;

		VkShaderModule vert_module = VK_NULL_HANDLE;

		VkShaderModule frag_module = VK_NULL_HANDLE;

		try {
			vert_module = createShaderModule("shaders/triangle.vert.spv");

			frag_module = createShaderModule("shaders/triangle.frag.spv");

			const VkPipelineShaderStageCreateInfo stages[] = {
			{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_VERTEX_BIT,
					.module = vert_module,
					.pName = "main",
				},
			{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
					.module = frag_module,
					.pName = "main",
				}
			};

			const VkVertexInputBindingDescription binding = makeVertexBindingDescription();

			const auto attributes = makeVertexPositionAttributeDescriptions();

			const VkPipelineVertexInputStateCreateInfo vertex_input = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &binding,
				.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
				.pVertexAttributeDescriptions = attributes.data(),
			};

			const VkPipelineInputAssemblyStateCreateInfo input_assembly = makeInputAssemblyState();

			const VkPipelineViewportStateCreateInfo viewport_state = makeViewportState();

			const VkDynamicState dynamic_states[] = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR,
			};

			const VkPipelineDynamicStateCreateInfo dynamic_state = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
				.dynamicStateCount = 2,
				.pDynamicStates = dynamic_states
			};

			const VkPipelineRasterizationStateCreateInfo rasterizer = makeRasterizationState();

			const VkPipelineMultisampleStateCreateInfo multisampling = makeMultisampleState();

			const VkPipelineDepthStencilStateCreateInfo depth_stencil = makeDepthStencilState();

			const VkPipelineColorBlendAttachmentState color_attachment = makeColorBlendAttachmentState();

			const VkPipelineColorBlendStateCreateInfo color_blending = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.attachmentCount = 1,
				.pAttachments = &color_attachment
			};

			const VkGraphicsPipelineCreateInfo pipeline_info = {
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.stageCount = 2,
				.pStages = stages,
				.pVertexInputState = &vertex_input,
				.pInputAssemblyState = &input_assembly,
				.pViewportState = &viewport_state,
				.pRasterizationState = &rasterizer,
				.pMultisampleState = &multisampling,
				.pDepthStencilState = &depth_stencil,
				.pColorBlendState = &color_blending,
				.pDynamicState = &dynamic_state,
				.layout = pipeline_layout,
				.renderPass = graphics::internal::context.render_pass,
				.subpass = 0,
			};

			if (vkCreateGraphicsPipelines(
				graphics::internal::context.device,
				VK_NULL_HANDLE, 1,
				&pipeline_info,
				nullptr,
				&result_pipeline
			) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create graphics pipeline");
			}
		} catch (const std::exception& e) {
			vkDestroyShaderModule(graphics::internal::context.device, vert_module, nullptr);
			vkDestroyShaderModule(graphics::internal::context.device, frag_module, nullptr);
			vkDestroyPipeline(graphics::internal::context.device, result_pipeline, nullptr);
			throw;
		}

		vkDestroyShaderModule(graphics::internal::context.device, vert_module, nullptr);
		vkDestroyShaderModule(graphics::internal::context.device, frag_module, nullptr);
		return result_pipeline;
	}

	glm::mat4 makeViewMatrix() {
		// lookAtRH строит преобразование в систему камеры; точка начала координат окажется перед ней с координатой z = -4
		const auto view_matrix = glm::lookAtRH(
			glm::vec3(0.0f, 0.0f, 4.0f), // положение камеры
			glm::vec3(0.0f, 0.0f, 0.0f), // точка, на которую смотрим
			glm::vec3(0.0f, 1.0f, 0.0f)  // направление вверх
		);
		return view_matrix;
	}

	glm::mat4 makeProjectionMatrix(float aspect, int mode) {
		glm::mat4 result(1.0f);
		// RH означает выбранную нами правую систему координат
		// ZO — глубину после деления в диапазоне
		switch(mode) {
			case 1 :
				result = glm::perspectiveRH_ZO(
					glm::radians(50.0f), // полный вертикальный угол обзора перспективной камеры
					aspect, // ширина изображения, делённая на высоту (соотношение сторон)
					// расстояния до ближней и дальней плоскостей отсечения, измеренные вдоль направления взгляда.
					0.1f,
					100.0f
				);
				break;
			case 2 :
				const float half_height = 1.8f; // задаёт ортографическую область высотой 3.6 единицы.
				// Ширину согласуем с пропорциями окна.
				const float half_width = half_height * aspect;

				result = glm::orthoRH_ZO(
					-half_width, half_width,
					-half_height, half_height,
					0.1f, 100.0f
				);
				break;
		}
		result[1][1] *= -1; // инвертируем ось Y, так как Vulkan использует верхнюю левую точку отсчета
		return result;
	}

	glm::vec3 makeAnimationOffset(float phase, float size) {
		const float envelope = 0.5f * (1.0f - glm::cos(phase));
    	const float max_amplitude = 0.6f * size;
		auto animation_offset = glm::vec3(
			size * glm::sin(phase),
			max_amplitude * envelope * glm::sin(4.0f * phase),
			0.15f * size * glm::sin(2.0f * phase)
    	);
		return animation_offset;
	}

	glm::mat4 makeModelMatrix(const ObjectState& state) {

		const glm::mat4 transformation = glm::translate(
			glm::mat4(1.0f),
			state.position
		);

		const glm::mat4 rotation = glm::rotate(
			glm::mat4(1.0f),
			glm::radians(state.rotation_x_degrees),
			glm::vec3(1.0f, 0.0f, 0.0f)
		);

		const glm::mat4 scaling = glm::scale(
			glm::mat4(1.0f),
			state.scale
		);

		return transformation * rotation * scaling;
	}

	void animateFigure(double time, ObjectState& state) {
		static double previous_time = time;
		auto& object_position = state.position;
		auto& rotation_x_degrees = state.rotation_x_degrees;

		const double delta_time = time - previous_time;
		previous_time = time;

		if (is_animate) {
			const float previous_phase = static_cast<float>(animation_phase);
			const double phase_step = animation_speed * delta_time;
			animation_phase += phase_step;
			const float phase = static_cast<float>(animation_phase);

			// Прибавляем только перемещение за этот кадр.
			object_position += makeAnimationOffset(phase, trajectory_size)
			                 - makeAnimationOffset(previous_phase, trajectory_size);
			rotation_x_degrees += glm::degrees(0.5f * static_cast<float>(phase_step));
			// Один оборот не меняет ориентацию; сохраняем угол в диапазоне ползунка.
			rotation_x_degrees = std::remainder(rotation_x_degrees, 360.0f);
		}


	}

}

bool initialize() {
	Mesh tour = makeTorus(0.8f, 0.35f, 8, 4);

	index_count = static_cast<uint32_t>(tour.indices.size());

	std::cout << "Vertices: " << tour.vertices.size() << std::endl;
	std::cout << "Indices: " << tour.indices.size() << std::endl;
	std::cout << "Index Count: " << index_count << std::endl;

	createMeshBuffers(tour);

	object_states[1].position = glm::vec3(1.6f, 0.0f, 0.0f);
	object_states[1].rotation_x_degrees = 0.0f;
	object_states[1].scale = glm::vec3(0.45f);
	object_states[1].color = glm::vec3(0.5f, 0.8f, 1.0f);	

	descriptor_set_layout = createDescriptorSetLayout();
	descriptor_pool = createDescriptorPool();
	for (auto& object : objects) {
		createUniformBuffer(object);
		object.descriptor_set = allocateDescriptorSet();

		writeUniformBufferDescriptor(object);

	}
	pipeline_layout = createPipelineLayout();
	pipeline =createGraphicsPipeline(pipeline_layout);

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);
	vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_allocation);
	vmaDestroyBuffer(context.allocator, index_buffer, index_allocation);
	for (auto& object : objects) {
		destroyUniformBuffer(object);
	}
	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
}

void update(double time) {
	ImGui::Begin("Torus");
	ImGui::RadioButton("Object 1", &selected_object, 0);
	ImGui::SameLine();
	ImGui::RadioButton("Object 2", &selected_object, 1);
	// Выбираем состояние после кнопок, чтобы переключение действовало в этом кадре.
	auto& current_object = object_states[selected_object];
	// Обновляем параметры до ползунков, чтобы они показывали текущий кадр.
	animateFigure(time, current_object);

	ImGui::SliderFloat("Rotation X", &current_object.rotation_x_degrees, -180.0f, 180.0f);
	ImGui::RadioButton("Perspective", &projection_mod, 1);
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", &projection_mod, 2);
	ImGui::Separator();
	ImGui::SliderFloat3(
		"Position", glm::value_ptr(current_object.position),
		-3.0f, 3.0f
	);
	ImGui::SliderFloat3(
		"Scale", glm::value_ptr(current_object.scale),
		0.1f, 3.0f, "%.2f",
		ImGuiSliderFlags_AlwaysClamp
	);
	ImGui::ColorEdit3(
		"Color", glm::value_ptr(current_object.color)
	);
	ImGui::Separator();
	ImGui::Checkbox("Animate", &is_animate);
	ImGui::SliderFloat(
		"Animation Speed", &animation_speed,
		 0.1f, 3.0f, "%.2f rad/s",
		 ImGuiSliderFlags_AlwaysClamp
	);
	ImGui::SliderFloat(
		"Trajectory Size", &trajectory_size,
		0.1f, 2.0f, "%.2f",
		ImGuiSliderFlags_AlwaysClamp
	);
	ImGui::End();

}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	VkClearValue clear_values[2]{};
	clear_values[0].color = { .float32 = { 0.08f, 0.12f, 0.20f, 1.0f } };
	clear_values[1].depthStencil = { 1.0f, 0 };


	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(
		fd.command_buffer,
		&render_pass_begin,
		VK_SUBPASS_CONTENTS_INLINE
	);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(context.swapchain_extent.width),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f
	};

	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = context.swapchain_extent
	};

	const VkDeviceSize offset = 0;

	const float aspect = 
		static_cast<float>(context.swapchain_extent.width) /
		static_cast<float>(context.swapchain_extent.height);

	const TransformUniforms transform = {
		.model = makeModelMatrix(object_states[0]),
		.view = makeViewMatrix(),
		.projection = makeProjectionMatrix(aspect, projection_mod),
		.color = glm::vec4(object_states[0].color, 1.0f)
	};

	const TransformUniforms second_transform = {
		.model = makeModelMatrix(object_states[1]),
		.view = makeViewMatrix(),
		.projection = makeProjectionMatrix(aspect, projection_mod),
		.color = glm::vec4(object_states[1].color, 1.0f)
	};

	std::vector<TransformUniforms> transform_uniforms = { transform, second_transform };

	for (const auto& transform : transform_uniforms) {
		updateUniformBuffer(transform, objects[&transform - &transform_uniforms[0]]);
	}

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	vkCmdBindVertexBuffers(
		fd.command_buffer,
		0,
		1,
		&vertex_buffer,
		&offset
	);
	vkCmdBindIndexBuffer(
		fd.command_buffer,
		index_buffer,
		0,
		VK_INDEX_TYPE_UINT32
	);

	for (const auto& object : objects) {
		vkCmdBindDescriptorSets(
			fd.command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipeline_layout,
			0, // первый набор привязываем в место set = 0
			1, &object.descriptor_set,
			0, nullptr // динамические смещения не используются
		);
		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
	}


	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
