mod compute;

use crate::compute::cpu::cpu_calc;
use crate::compute::gpu::GpuContext;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup Phase
    // We'll test with 10 million elements
    let count = 10_000_000;
    let gpu = GpuContext::new(count)?;
    let input_data = vec![10u32; count];

    println!("🚀 Gateway Engine Initialized (Batch Size: {})", count);
    println!("--------------------------------------------------");

    // --- GPU BENCHMARK ---
    println!("GPU: Starting pipeline (Write -> Compute -> Read)...");
    let gpu_start = Instant::now();

    // Step A: Upload
    gpu.write_to_buffer(&input_data)?;

    // Step B: Compute
    gpu.run_compute()?;

    // Step C: Download
    let mut gpu_output = vec![0u32; count];
    gpu.read_from_buffer(&mut gpu_output)?;

    let gpu_duration = gpu_start.elapsed();
    println!("GPU Pipeline Finished in: {:?}", gpu_duration);

    // --- CPU BENCHMARK ---
    println!("CPU: Starting parallel compute (Rayon)...");
    let mut cpu_data = input_data.clone();

    let cpu_start = Instant::now();
    cpu_calc(&mut cpu_data);
    let cpu_duration = cpu_start.elapsed();

    println!("CPU Pipeline Finished in: {:?}", cpu_duration);
    println!("--------------------------------------------------");

    // --- VERIFICATION ---
    let expected_val = 20u32;
    let gpu_success = gpu_output[0] == expected_val && gpu_output[count - 1] == expected_val;
    let cpu_success = cpu_data[0] == expected_val && cpu_data[count - 1] == expected_val;

    if gpu_success && cpu_success {
        println!("✅ DATA INTEGRITY VERIFIED");
    } else {
        println!("❌ DATA ERROR: Results do not match expected doubling");
        if !gpu_success { println!("   (Check gpu.rs for double-dispatch bug)"); }
    }

    // --- ANALYSIS ---
    if gpu_duration < cpu_duration {
        println!("🏆 WINNER: GPU is faster by {:?}", cpu_duration - gpu_duration);
    } else {
        println!("🏆 WINNER: CPU is faster by {:?}", gpu_duration - cpu_duration);
    }

    Ok(())
}