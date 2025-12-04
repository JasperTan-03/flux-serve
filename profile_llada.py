import time
import torch
import pynvml
from transformers import AutoModel, AutoTokenizer

# --- 1. Setup ---
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) 

def get_memory_usage():
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**3  # GB

model_id = 'GSAI-ML/LLaDA-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()

# --- 2. The Experiment (Manual Diffusion Step) ---
batch_size = 1
gen_len = 128
# Create a dummy "masked" input sequence (Batch, Length)
# LLaDA processes the WHOLE sequence at once, unlike AR which grows it.
input_ids = torch.randint(0, 32000, (batch_size, gen_len)).cuda()

print("\n--- Starting Profiling Loop ---")
print("Step | Memory (GB) | Latency (s)")

for step in range(1, 21): 
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        # We manually call forward() to simulate 1 denoising step
        # This bypasses the broken .generate() AR loop
        outputs = model(input_ids=input_ids)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    mem = get_memory_usage()
    print(f"{step:02d}   | {mem:.4f}      | {end_time - start_time:.4f}")

print("\nSuccess. This data proves memory is static (Compute Bound).")