use ash::vk;
use std::ffi::CStr;

pub struct GpuContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family_index: u32,

    pub buffer: vk::Buffer,
    pub buffer_memory: vk::DeviceMemory,
    pub shader_module: vk::ShaderModule,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub max_elements: usize,
}

impl GpuContext {
    pub fn new(max_elements: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { ash::Entry::load()? };

        let extension_names = [
            ash::khr::portability_enumeration::NAME.as_ptr(),
            ash::khr::get_physical_device_properties2::NAME.as_ptr(),
        ];
        let layer_names = [CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0")?.as_ptr()];

        let create_info = vk::InstanceCreateInfo::default()
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names)
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        let physical_device = unsafe { instance.enumerate_physical_devices()? }[0];
        let queue_family_index = unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .expect("No compute queue") as u32;

        let device_extensions = [CStr::from_bytes_with_nul(b"VK_KHR_portability_subset\0")?.as_ptr()];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&[1.0f32]);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let buffer_size = (max_elements * 4) as u64;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let mem_type_index = (0..mem_props.memory_type_count)
            .find(|i| {
                (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
                    mem_props.memory_types[*i as usize].property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                    )
            }).unwrap();

        let buffer_memory = unsafe {
            device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(mem_reqs.size).memory_type_index(mem_type_index), None)?
        };
        unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0)? };

        let shader_binary = include_bytes!("../shaders/handshake.spv");
        let shader_code = ash::util::read_spv(&mut std::io::Cursor::new(shader_binary))?;
        let shader_module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&shader_code), None)? };

        let layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&layout_binding)), None)? };
        let pipeline_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&descriptor_set_layout)), None)? };

        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[
                vk::ComputePipelineCreateInfo::default()
                    .stage(vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader_module)
                        .name(CStr::from_bytes_with_nul(b"main\0")?))
                    .layout(pipeline_layout)
            ], None).map_err(|e| e.1)?[0]
        };

        let descriptor_pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)]), None)? };
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&descriptor_set_layout)))?[0] };

        unsafe {
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk::DescriptorBufferInfo::default().buffer(buffer).offset(0).range(buffer_size)])], &[]);
        }

        Ok(Self {
            entry, instance, device, physical_device, queue, queue_family_index,
            buffer, buffer_memory, shader_module, descriptor_set_layout,
            pipeline_layout, pipeline, descriptor_pool, descriptor_set,
            max_elements,

        })
    }

    pub fn write_to_buffer(&self, input_data: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
        let size = (input_data.len() * std::mem::size_of::<u32>()) as u64;
        unsafe {
            let ptr = self.device.map_memory(self.buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(input_data.as_ptr(), ptr as *mut u32, input_data.len());
            self.device.unmap_memory(self.buffer_memory);
        }
        Ok(())
    }

    pub fn read_from_buffer(&self, output_data: &mut [u32]) -> Result<(), Box<dyn std::error::Error>> {
        let size = (output_data.len() * std::mem::size_of::<u32>()) as u64;
        unsafe {
            let ptr = self.device.map_memory(self.buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(ptr as *const u32, output_data.as_mut_ptr(), output_data.len());
            self.device.unmap_memory(self.buffer_memory);
        }
        Ok(())
    }

    pub fn run_compute(&self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            // 1. Setup Command Pool
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER); // Corrected for ash 0.38

            let command_pool = self.device.create_command_pool(&pool_info, None)?;

            // 2. Allocate Command Buffer
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer = self.device.allocate_command_buffers(&alloc_info)?[0];

            // 3. Record Commands
            self.device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.descriptor_set], &[]);

            // calculate groups: (total items / 64
            let group_count = (self.max_elements as u32 + 63) /64;

            // only one dispatch call is needed, and it must be recorded here:
            self.device.cmd_dispatch(command_buffer, group_count, 1,1);
            
            self.device.end_command_buffer(command_buffer)?;


            // 4. Submit to Queue
            let command_buffers_to_submit = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers_to_submit);
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            self.device.queue_submit(self.queue, &[submit_info], fence)?;

            // 5. Wait for GPU to finish
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;

            // 6. Local Cleanup
            self.device.destroy_fence(fence, None);
            self.device.destroy_command_pool(command_pool, None);
        }
        Ok(())
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            println!("GpuContext: Cleaning up resources...");
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device.free_memory(self.buffer_memory, None);
            self.device.destroy_buffer(self.buffer, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}