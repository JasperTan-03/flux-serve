import time
import torch
import queue
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer

# --- Configuration ---
MODEL_ID = 'GSAI-ML/LLaDA-8B-Instruct'
MAX_BATCH_SIZE = 32  # From your profiling
STEPS_PER_REQ = 20   # LLaDA default
SEQ_LEN = 128        # Fixed sequence length

# Set HuggingFace cache to /work directory to avoid home quota issues
# /work has much more space available (3.9P vs 10GB quota on /home1)
hf_cache_dir = os.environ.get('HF_HOME', '/work/09889/jaspertan03/.cache/huggingface')
os.environ['HF_HOME'] = hf_cache_dir
os.makedirs(hf_cache_dir, exist_ok=True)
print(f"Using HuggingFace cache directory: {hf_cache_dir}")

# --- 1. Load Real Model ---
print("Loading LLaDA (Real System)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=hf_cache_dir)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=hf_cache_dir).cuda()
model.eval()
print("Model Loaded.")

# --- 2. The Dynamic Scheduler Logic (Real Implementation) ---
def run_dynamic_batching_demo(requests_to_process):
    """
    Simulates a queue of requests arriving.
    Instead of math, we actually run model.forward().
    """
    request_queue = queue.Queue()
    for r in requests_to_process:
        request_queue.put(r)
        
    completed_requests = []
    
    print(f"\nProcessing {request_queue.qsize()} requests on Real GPU...")
    start_wall_time = time.time()
    
    while not request_queue.empty():
        # --- SCHEDULING DECISION ---
        # Greedy strategy: Pull as many as we can fit (up to MAX_BATCH)
        batch_items = []
        while not request_queue.empty() and len(batch_items) < MAX_BATCH_SIZE:
            batch_items.append(request_queue.get())
            
        current_bs = len(batch_items)
        
        # --- EXECUTION (Real GPU Work) ---
        step_start = time.time()
        
        # Prepare inputs (We use dummy tensors for speed, but shape is real)
        # In a real chat app, you'd tokenize the actual prompts here.
        input_ids = torch.randint(0, 32000, (current_bs, SEQ_LEN)).cuda()
        
        with torch.no_grad():
            # Run the specific number of diffusion steps
            for _ in range(STEPS_PER_REQ):
                _ = model(input_ids=input_ids)
                
        torch.cuda.synchronize() # Wait for GPU to finish
        step_end = time.time()
        
        actual_latency = step_end - step_start
        
        # Record stats
        print(f"Processed Batch of {current_bs} | Time: {actual_latency:.4f}s")
        completed_requests.extend([actual_latency] * current_bs)

    total_time = time.time() - start_wall_time
    return total_time, completed_requests

# --- 3. Run the Comparison ---
if __name__ == "__main__":
    # Create a "Burst" workload of 50 requests
    # In a real system, these would arrive over time. 
    # Here we assume they are already in the queue (Saturation Test).
    N_REQUESTS = 50
    fake_requests = range(N_REQUESTS)
    
    print(f"--- VALIDATION EXPERIMENT: {N_REQUESTS} Requests ---")
    
    # Run Real System
    real_duration, real_latencies = run_dynamic_batching_demo(fake_requests)
    avg_real = sum(real_latencies) / len(real_latencies)
    
    # Run "Mental" Simulation (using your profile data)
    # Batch 32 takes ~0.40s, Batch 18 takes ~0.24s (interpolated)
    # 50 requests -> Batch 32 (0.40s) + Batch 18 (0.24s) = ~0.64s total (Theoretical)
    # *Note: This is rough math. Your simulator does this precisely.*
    
    print("\n--- RESULTS ---")
    print(f"Total Wall Time: {real_duration:.4f}s")
    print(f"Average Request Latency: {avg_real:.4f}s")
    
    print("\nCompare this 'Average Request Latency' to your Simulator's output for a similar burst.")
    print("If they are close, your simulation is VALID.")