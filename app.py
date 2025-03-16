import torch
import nvcomp
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Llama 3 Model and Tokenizer
model_name = "your-llama3-model-path"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Ensure model weights are contiguous before compression
for param in model.parameters():
    param.data = param.data.contiguous().to(dtype=torch.float16)

# ------------------------------
# 1. Compress Model Weights Using nvcomp
# ------------------------------

def compress_tensor_gpu(tensor, algorithm="LZ4"):
    """Compress tensor using GPU-accelerated nvcomp"""
    tensor_data = tensor.cpu().numpy()  # Move to CPU for compression
    compressed_data, temp_space, output_size = nvcomp.compress(tensor_data, algorithm=algorithm)
    return compressed_data, output_size

# Compress all model parameters
compressed_weights = []
original_shapes = []

for param in model.parameters():
    compressed_param, size = compress_tensor_gpu(param.data)
    compressed_weights.append(compressed_param)
    original_shapes.append(param.data.shape)

print(f"Model Weights Compressed. Total {len(compressed_weights)} tensors.")

# ------------------------------
# 2. Decompress Model Weights for Inference
# ------------------------------

def decompress_tensor_gpu(compressed_data, output_shape, algorithm="LZ4"):
    """Decompress tensor directly on the GPU"""
    decompressed_data = nvcomp.decompress(compressed_data, output_shape, algorithm=algorithm)
    return torch.tensor(decompressed_data, dtype=torch.float16).cuda()

# Restore model weights before inference
for i, param in enumerate(model.parameters()):
    param.data = decompress_tensor_gpu(compressed_weights[i], original_shapes[i])

print("Model Weights Decompressed and Loaded.")

# ------------------------------
# 3. Compress KV Cache in Real-Time
# ------------------------------

def compress_kv_cache(kv_cache):
    """Compress KV cache tensors using nvcomp"""
    compressed_cache = {}
    for layer in kv_cache:
        key, value = layer
        key_comp, key_size = compress_tensor_gpu(key)
        value_comp, value_size = compress_tensor_gpu(value)
        compressed_cache[layer] = (key_comp, key_size, value_comp, value_size)
    return compressed_cache

def decompress_kv_cache(compressed_cache):
    """Decompress KV cache tensors using nvcomp"""
    decompressed_cache = []
    for key_comp, key_size, value_comp, value_size in compressed_cache.values():
        key_dec = decompress_tensor_gpu(key_comp, key_size)
        value_dec = decompress_tensor_gpu(value_comp, value_size)
        decompressed_cache.append((key_dec, value_dec))
    return decompressed_cache

# ------------------------------
# 4. Inference Function with On-the-Fly KV Cache Decompression
# ------------------------------

def generate_response(prompt, compressed_kv_cache=None):
    """Generate response with compressed KV cache"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # If KV cache is available, decompress before inference
    if compressed_kv_cache is not None:
        kv_cache = decompress_kv_cache(compressed_kv_cache)
    else:
        kv_cache = None  # First inference run

    with torch.no_grad():
        outputs = model.generate(input_ids, use_cache=True, past_key_values=kv_cache)

    # Compress KV cache after inference for reuse
    new_kv_cache = model.get_cache()
    compressed_kv_cache = compress_kv_cache(new_kv_cache)

    return tokenizer.decode(outputs[0], skip_special_tokens=True), compressed_kv_cache

# ------------------------------
# 5. Run Inference with KV Cache Compression
# ------------------------------

# First run - no KV cache available
response, compressed_kv_cache = generate_response("Hello, how are you?")
print(response)

# Second run - reuse compressed KV cache
response, compressed_kv_cache = generate_response("Tell me a joke.", compressed_kv_cache)
print(response)

