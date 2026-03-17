## How do the Paged Attention(PA) saves memory, particularly in case of large language models
# Understanding PagedAttention in Large Language Models

In traditional Large Language Model (LLM) serving, memory management for the **Key-Value (KV) Cache** is the primary bottleneck. **PagedAttention**, the core algorithm behind the vLLM engine, solves this by applying the concept of **Virtual Memory** from operating systems to the GPU's memory.

## Key Mechanisms for Memory Efficiency

### 1. Eliminating "Internal Fragmentation"
In standard systems (like Hugging Face Transformers), memory for the KV cache of a request is allocated as a single, contiguous block. To handle a potential output of 512 tokens, the system must pre-allocate space for all 512 tokens immediately, even if the model only ends up generating 10.

* **The Waste:** This "reserved but unused" space cannot be used by other requests.
* **The Paged Solution:** PagedAttention divides the KV cache into fixed-size **blocks** (e.g., 16 tokens). Memory is only allocated when a block is filled. If a request needs 17 tokens, it only takes two blocks, rather than a giant pre-allocated chunk.

### 2. Eliminating "External Fragmentation"
Because requests vary in length, contiguous allocation often leaves small "holes" of memory between active requests that are too small for a new request to fit into, even if the total free memory is high.

* **The Paged Solution:** Since PagedAttention treats blocks like pages in an OS, they don't need to be physically adjacent. The system uses a **block table** to map logical tokens to non-contiguous physical memory locations. This allows the engine to utilize almost 100% of the available GPU VRAM.

### 3. Efficient Memory Sharing
One of the most powerful features of PagedAttention is how it handles complex sampling (like parallel sampling or beam search) where multiple output sequences share the same starting prompt.

* **Standard Way:** The system would duplicate the KV cache for the prompt for every single output sequence.
* **The Paged Solution:** Multiple sequences can point to the same physical memory blocks for the initial prompt. The system only starts "copying-on-write" at the block level once the sequences begin to diverge. This can reduce memory usage by up to **55%** in scenarios with heavy branching.

---

## Comparison Summary

| Feature | Traditional KV Cache | PagedAttention |
| :--- | :--- | :--- |
| **Allocation** | Static & Contiguous | Dynamic & Block-based |
| **Utilization** | Often < 40% due to over-reservation | Near 100% |
| **Fragmentation** | High (Internal & External) | Negligible |
| **Shared Prompts** | Duplicated in memory | Shared via Block Table |

## Impact
By moving from contiguous allocation to this "paged" approach, LLM serving engines can increase their throughput by **2x to 4x** on the same hardware because they can fit many more concurrent requests into the GPU.
