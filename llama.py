import torch
from nvidia import nvcomp
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# ------------------------------
# 0. Load Model and Tokenizer
# ------------------------------

# Hugging Face model configuration
#MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID = "meta-llama/Llama-3.2-1B"

def load_model_from_hf(model_id, use_auth=True):
    """
    Load model from Hugging Face, handling authentication if needed
    """
    print(f"Loading {model_id} from Hugging Face...")
    
    if use_auth:
        # Check if HF_TOKEN is set in environment variables
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            # Prompt for token if not in environment
            print(f"Hugging Face authentication token required for {model_id}")
            print("Get your token at https://huggingface.co/settings/tokens")
            hf_token = input("Enter your Hugging Face token: ")
        
        # Login to Hugging Face
        login(token=hf_token)
        print("Successfully logged in to Hugging Face")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model
    print("Loading model (this may take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        device_map="auto",          # Automatically distribute across available GPUs
        low_cpu_mem_usage=True      # Optimize memory usage during loading
    )
    
    print(f"Successfully loaded {model_id}")
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model_from_hf(MODEL_ID)

# Ensure model weights are contiguous before compression
for param in model.parameters():
    param.data = param.data.contiguous().to(dtype=torch.float16)

# ------------------------------
# 1. Compress Model Weights Using nvcomp
# ------------------------------

def compress_tensor_gpu(tensor, algorithm="LZ4"):
    """Compress tensor using GPU-accelerated nvcomp"""
    # Ensure tensor is on GPU and contiguous
    tensor_data = tensor.contiguous().cuda()
    
    # Create nvcomp codec for the specified algorithm
    codec = nvcomp.Codec(algorithm=algorithm)
    
    # Wrap tensor in nvcomp Array
    nv_tensor = nvcomp.as_array(tensor_data)
    
    # Compress the data
    compressed_data = codec.encode([nv_tensor])[0]
    
    return compressed_data, tensor_data.shape

# Compress all model parameters
compressed_weights = []
original_shapes = []

print("Compressing model weights...")
for param in model.parameters():
    compressed_param, shape = compress_tensor_gpu(param.data)
    compressed_weights.append(compressed_param)
    original_shapes.append(shape)

print(f"Model Weights Compressed. Total {len(compressed_weights)} tensors.")

# ------------------------------
# 2. Decompress Model Weights for Inference
# ------------------------------

def decompress_tensor_gpu(compressed_data, output_shape, algorithm="LZ4"):
    """Decompress tensor directly on the GPU"""
    # Create nvcomp codec
    codec = nvcomp.Codec(algorithm=algorithm)
    
    # Decompress the data
    decompressed_data = codec.decode([compressed_data])[0]
    
    # Convert back to torch tensor with the right shape and dtype
    # Ensure we're using float16 for model parameters
    return torch.as_tensor(decompressed_data, device="cuda", dtype=torch.float16).view(output_shape)

# Create a function to stream decompress only the needed layers during inference
def stream_decompress_layers(layer_indices):
    """Decompress only specific layers of the model on demand"""
    param_index = 0
    
    for module_name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if param_index in layer_indices:
                module.weight.data = decompress_tensor_gpu(
                    compressed_weights[param_index], 
                    original_shapes[param_index]
                )
            param_index += 1
        
        if hasattr(module, 'bias') and module.bias is not None:
            if param_index in layer_indices:
                module.bias.data = decompress_tensor_gpu(
                    compressed_weights[param_index], 
                    original_shapes[param_index]
                )
            param_index += 1

# ------------------------------
# 3. Compress KV Cache in Real-Time
# ------------------------------

def compress_kv_cache(kv_cache):
    """Compress KV cache tensors using nvcomp directly on GPU"""
    compressed_cache = {}
    codec = nvcomp.Codec(algorithm="LZ4")
    
    for layer_idx, layer in enumerate(kv_cache):
        key, value = layer
        # Ensure tensors are contiguous and on GPU
        key_gpu = key.contiguous().cuda()
        value_gpu = value.contiguous().cuda()
        
        # Wrap in nvcomp arrays
        key_nv = nvcomp.as_array(key_gpu)
        value_nv = nvcomp.as_array(value_gpu)
        
        # Compress
        key_comp = codec.encode([key_nv])[0]
        value_comp = codec.encode([value_nv])[0]
        
        compressed_cache[layer_idx] = (key_comp, key.shape, value_comp, value.shape)
    
    return compressed_cache
def decompress_kv_cache(compressed_cache):
    """Decompress KV cache tensors using nvcomp directly on GPU"""
    decompressed_cache = []
    codec = nvcomp.Codec(algorithm="LZ4")
    
    for layer_idx in sorted(compressed_cache.keys()):
        key_comp, key_shape, value_comp, value_shape = compressed_cache[layer_idx]
        
        # Decompress directly on GPU and convert to PyTorch tensor
        key_dec_data = codec.decode([key_comp])[0]
        value_dec_data = codec.decode([value_comp])[0]
        
        # Convert to PyTorch tensors and reshape with correct dtype
        key_dec = torch.as_tensor(key_dec_data, device="cuda", dtype=torch.float16).view(key_shape)
        value_dec = torch.as_tensor(value_dec_data, device="cuda", dtype=torch.float16).view(value_shape)
        
        decompressed_cache.append((key_dec, value_dec))
    
    return decompressed_cache
    return decompressed_cache

# ------------------------------
# 4. Inference Function with On-the-Fly Streaming Decompression
# ------------------------------

def generate_response(prompt, compressed_kv_cache=None, active_layers=None):
    """
    Generate response with compressed KV cache and streaming layer decompression
    
    Args:
        prompt: Input text
        compressed_kv_cache: Previously compressed KV cache (optional)
        active_layers: List of layer indices to decompress (if None, decompress all)
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    
    # Determine which layers to decompress
    if active_layers is None:
        # Default: decompress all layers for full inference
        active_layers = list(range(len(compressed_weights)))
    
    # Stream decompress only the needed layers
    stream_decompress_layers(active_layers)
    
    # If KV cache is available, decompress before inference
    kv_cache = None
    if compressed_kv_cache is not None:
        kv_cache = decompress_kv_cache(compressed_kv_cache)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            use_cache=True,
            past_key_values=kv_cache,
            max_new_tokens=100
        )
    
    # Compress KV cache after inference for reuse
    new_kv_cache = None
    if hasattr(model, 'get_cache') and callable(model.get_cache):
        new_kv_cache = model.get_cache()
        compressed_kv_cache = compress_kv_cache(new_kv_cache)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True), compressed_kv_cache

# ------------------------------
# 5. Run Inference with Streaming Decompression and KV Cache Compression
# ------------------------------

# First run - decompress all layers, no KV cache available
print("\nGenerating first response (full model)...")
response, compressed_kv_cache = generate_response("Hello, how are you?")
print(response)

# Second run - reuse compressed KV cache, and optionally decompress only specific layers
print("\nGenerating second response (with compressed KV cache)...")
response, compressed_kv_cache = generate_response(
    "Tell me a joke.", 
    compressed_kv_cache=compressed_kv_cache
)
print(response)

# Example of streaming only specific layers (e.g., only the first 10 layers)
# This is just an example and may need to be adjusted based on your model architecture
print("\nGenerating third response (with partial model decompression)...")
response, compressed_kv_cache = generate_response(
    "What's the weather like?", 
    compressed_kv_cache=compressed_kv_cache,
    active_layers=list(range(10))  # Only decompress first 10 parameters
)
print(response)

