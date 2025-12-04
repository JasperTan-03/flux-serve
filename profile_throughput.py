import time
import torch
import pynvml
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

# --- Setup ---
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) 

model_id = 'GSAI-ML/LLaDA-8B-Instruct'
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()

# --- The Experiment: Batch Size Scaling ---
# We want to see if doubling the batch size doubles the time (Linear Scaling)
# or if it takes less than double (Sub-linear Scaling).
# Sub-linear means we have "free compute" to exploit!

batch_sizes = [1, 2, 4, 8, 16, 32]
seq_len = 128
results = []

print("\n--- Starting Throughput Profiling ---")
print(f"{'Batch':<6} | {'Latency (s)':<12} | {'Throughput (req/s)':<20} | {'Memory (GB)':<12}")

for b in batch_sizes:
    # 1. Prepare Input
    input_ids = torch.randint(0, 32000, (b, seq_len)).cuda()
    
    # 2. Warmup (critical for accurate timing)
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_ids=input_ids)
    torch.cuda.synchronize()

    # 3. Profile
    start_time = time.time()
    iterations = 10
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(input_ids=input_ids)
    torch.cuda.synchronize()
    
    # 4. Calculate Stats
    total_time = time.time() - start_time
    avg_latency = total_time / iterations
    throughput = b / avg_latency
    
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_gb = mem_info.used / 1024**3
    
    print(f"{b:<6} | {avg_latency:<12.4f} | {throughput:<20.2f} | {mem_gb:<12.2f}")
    results.append((b, avg_latency, throughput))

print("\nDone. Plot 'Batch Size' (X) vs 'Throughput' (Y).")