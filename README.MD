### 1. The License Question (MIT vs. Monetization)
*   **The MIT License:** This is the "Gold Standard" for Rust portfolio projects. It says, "You can use this for anything, but I'm not liable if it breaks."
    *   **Pros:** It makes recruiters much more likely to look at your code because they don't have to worry about legal issues.
    *   **Monetization:** You **can** still monetize MIT code. You can sell a hosted version (SaaS), provide support, or sell a "Pro" version with extra features.
*   **Alternative (AGPL):** If you are terrified of a big company like Amazon taking your code and running it without paying you, you would use **AGPL**. But this often scares away recruiters and collaborators.
*   **My Advice:** Stick with **MIT**. For this project, your "Product" is your **expertise**. The goal is to get a high-paying job. The value of the job is much higher than the value of the code right now.

---

### 2. The Updated README.md


```markdown
# 🚀 Adaptive Secure Gateway

> **🚧 Project Status: Work in Progress (Alpha)**  
> Currently implementing the Adaptive Scheduler. The Vulkan Compute Pipeline and CPU Benchmark Harness are verified on macOS (Apple Silicon via MoltenVK) and Linux.

A high-performance Rust gateway that optimizes secure connection handshakes using adaptive CPU/GPU execution (Vulkan compute).

---

## 📖 Overview
**Adaptive Secure Gateway** is a systems-focused project exploring how to efficiently handle large volumes of secure connection handshakes under load. 

Handshaking (the cryptographic setup phase) is computationally expensive. This gateway acts as a drop-in proxy that offloads that setup, dynamically choosing the most efficient execution path:
*   **CPU (SIMD/Rayon):** Optimized for low-latency, small-batch workloads.
*   **GPU (Vulkan Compute):** Optimized for high-throughput, large-batch processing during traffic spikes.

### The Core Research Question:
*At what specific batch size does the overhead of GPU data-transfer (PCIe/Bus latency) become smaller than the time saved by massive parallelization?*

---

## 🧠 Motivation
Modern CDNs, APIs, and messaging platforms must handle thousands of handshakes per second. During a traffic spike or a DDoS event, the handshake phase becomes a bottleneck. This project investigates:
*   Heterogeneous compute strategies (CPU + GPU).
*   Batching strategies for cryptographic primitives.
*   Throughput vs. Latency trade-offs in real-time systems.

---

## ⚙️ How It Works
```text
Client Requests
      ↓
[ Adaptive Gateway ]
      ├── Batch Queue (Collects incoming handshakes)
      ├── Scheduler (Decides CPU vs. GPU based on batch size)
      ├── Compute Engine (Vulkan or Rayon)
      ↓
Backend Service (Forwarded secure session)
```

**Workflow:**
1. Incoming handshake requests are queued.
2. The **Scheduler** checks the current queue depth.
3. If the batch is small, it uses the **CPU (Rayon)** to minimize latency.
4. If the batch is large, it "mails" the work to the **GPU (Vulkan)** for throughput.
5. Results are mapped back to RAM and the sessions are established.

---

## 🔥 Features
*   **⚡ Adaptive Scheduling:** (In Progress) Dynamic path selection based on real-time load.
*   **🧮 Vulkan Compute Pipeline:** Custom compute shaders for parallel processing on Apple Silicon and Linux.
*   **📦 Manual Memory Management:** Direct control over GPU Buffer-to-Memory mapping for maximum performance.
*   **📊 Benchmarking Harness:** Integrated timers to measure the "Crossover Point" between CPU and GPU execution.

---

## 🏗️ Project Structure
```text
src/
├── main.rs            # Entry point & Benchmark harness
├── compute/
│   ├── mod.rs         # Compute module definitions
│   ├── cpu.rs         # Rayon-based parallel CPU execution
│   └── gpu.rs         # Ash-based Vulkan GPU execution
├── shaders/
│   ├── handshake.comp # GLSL Compute Shader source
│   └── handshake.spv  # Compiled SPIR-V binary (embedded in Rust)
└── scheduler/         # (WIP) Logic for execution path selection
```

---

## 🚀 Getting Started

### Prerequisites
*   **Rust:** Latest stable version.
*   **Vulkan SDK:** Installed and in your system PATH.
*   **Shader Compiler:** `glslangValidator` (included in Vulkan SDK).

### Build & Run
1. Compile the shaders:
   ```bash
   glslangValidator -V src/shaders/handshake.comp -o src/shaders/handshake.spv
   ```
2. Run the benchmarks:
   ```bash
   cargo run --release
   ```

---

## 📊 Roadmap
- [x] Minimal Vulkan compute pipeline (Instance, Device, Queue setup).
- [x] Memory mapping and Two-Way data bridge (Write/Read).
- [x] CPU vs GPU benchmark harness.
- [ ] Adaptive scheduler (Dynamic crossover point detection).
- [ ] Request Batching System (Queue management).
- [ ] Post-Quantum Cryptography (ML-KEM/Kyber) implementation.
- [ ] Gateway API layer (Proxying real TCP/TLS traffic).

---

## 📜 License
This project is licensed under the **MIT License**. See `LICENSE` for details.
MIT License
────────────────────────────────────────────────────────────
S W A R E
S C O P E
S T U D I O
2 0 2 6

███████╗██╗    ██╗ █████╗ ██████╗ ███████╗    
██╔════╝██║    ██║██╔══██╗██╔══██╗██╔════╝    
███████╗██║ █╗ ██║███████║██████╔╝█████╗      
╚════██║██║███╗██║██╔══██║██╔══██╗██╔══╝      
███████║╚███╔███╔╝██║  ██║██║  ██║███████╗    
╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝

███████╗ ██████╗ ██████╗ ██████╗ ███████╗     
██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝     
███████╗██║     ██║   ██║██████╔╝█████╗       
╚════██║██║     ██║   ██║██╔═══╝ ██╔══╝       
███████║╚██████╗╚██████╔╝██║     ███████╗     
╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝

███████╗████████╗██╗   ██╗██████╗ ██╗ ██████╗
██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║██╔═══██╗
███████╗   ██║   ██║   ██║██║  ██║██║██║   ██║
╚════██║   ██║   ██║   ██║██║  ██║██║██║   ██║
███████║   ██║   ╚██████╔╝██████╔╝██║╚██████╔╝
╚══════╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝


        MAHAMMAD CARLTON — 2026

GPU COMPUTE · SYSTEMS · VULKAN · RUST

“ENGINEERED AT THE EDGE OF THE MACHINE”
────────────────────────────────────────────────────────────

```
