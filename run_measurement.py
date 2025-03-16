import torch
import os
import zlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Check if nvcomp is available
try:
    from nvidia import nvcomp
    NVCOMP_AVAILABLE = True
    print("NVIDIA nvcomp library available")
except ImportError:
    NVCOMP_AVAILABLE = False
    print("NVIDIA nvcomp library not available, falling back to CPU compression")

# ------------------------------
# 0. Load Model and Tokenizer
# ------------------------------

# Use the specified model
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

def load_model_from_hf(model_id):
    """Load model from Hugging Face"""
    print(f"Loading {model_id} from Hugging Face...")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    print("Loading model (this may take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
"run_measurement.py" [noeol] 266L, 8681B                                                                             1,1           Top
hsatgnode3@3b608c30-01:~$ ls
KernelBench  hackathon  llama3-kvcache-compress
hsatgnode3@3b608c30-01:~$ cp ~/hackathon/team_gpu0/llama3-kvcache-compress/new_qwen.py llama3-kvcache-compress/run_measurement.py
hsatgnode3@3b608c30-01:~$ cd llama3-kvcache-compress/
hsatgnode3@3b608c30-01:~/llama3-kvcache-compress$ ls
README.md  app.py  llama.py  qwen.py  run_measurement.py
hsatgnode3@3b608c30-01:~/llama3-kvcache-compress$ git status
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	run_measurement.py

nothing added to commit but untracked files present (use "git add" to track)
hsatgnode3@3b608c30-01:~/llama3-kvcache-compress$ git add run_measurement.py
hsatgnode3@3b608c30-01:~/llama3-kvcache-compress$ git commit -m "measurement"
[main e547575] measurement
 Committer: Semianalysis Hackathon Node 3 <hsatgnode3@3b608c30-01.cloud.together.ai>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly. Run the
following command and follow the instructions in your editor to edit
your configuration file:

    git config --global --edit

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 1 file changed, 266 insertions(+)
 create mode 100644 run_measurement.py
hsatgnode3@3b608c30-01:~/llama3-kvcache-compress$ ls
README.md  app.py  llama.py  qwen.py  run_measurement.py
hsatgnode3@3b608c30-01:~/llama3-kvcache-compress$ vi run_measurement.py
hsatgnode3@3b608c30-01:~/llama3-kvcache-compress$ cat run_measurement.py
import torch
import os
import zlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Check if nvcomp is available
try:
    from nvidia import nvcomp
    NVCOMP_AVAILABLE = True
    print("NVIDIA nvcomp library available")
except ImportError:
    NVCOMP_AVAILABLE = False
    print("NVIDIA nvcomp library not available, falling back to CPU compression")

# ------------------------------
# 0. Load Model and Tokenizer
# ------------------------------

# Use the specified model
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

def load_model_from_hf(model_id):
    """Load model from Hugging Face"""
    print(f"Loading {model_id} from Hugging Face...")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    print("Loading model (this may take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )

    # Explicitly move to first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Successfully loaded {model_id} on {device}")
    return model, tokenizer, device

# ------------------------------
# Helper Functions
# ------------------------------

def measure_memory_usage():
    """Measure and report GPU memory usage"""
    if torch.cuda.is_available():
        # Calculate memory usage
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB

        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
    else:
        print("\nGPU not available for memory measurement")

# ------------------------------
# 1. Compress and Decompress Model Weights
# ------------------------------

def compress_tensor(tensor):
    """Compress tensor using zlib"""
    try:
        # Convert tensor to bytes
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()

        # Compress bytes
        compressed = zlib.compress(tensor_bytes)

        # Return with metadata
        return {
            'data': compressed,
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'original_size': len(tensor_bytes),
            'compressed_size': len(compressed)
        }
    except Exception as e:
        print(f"Compression error: {e}")
        return None

def decompress_tensor(compressed_obj, device):
    """Decompress tensor from zlib compressed bytes"""
    try:
        # Extract data and metadata
        compressed_data = compressed_obj['data']
        shape = compressed_obj['shape']
        dtype = compressed_obj['dtype']

        # Decompress
        decompressed = zlib.decompress(compressed_data)

        # Reconstruct tensor
        flat_tensor = torch.frombuffer(decompressed, dtype=dtype)

        # Verify size
        expected_elements = 1
        for dim in shape:
            expected_elements *= dim

        if flat_tensor.numel() != expected_elements:
            print(f"WARNING: Size mismatch! Expected {expected_elements} elements, got {flat_tensor.numel()}")
            # Handle mismatch by truncating or padding
            if flat_tensor.numel() > expected_elements:
                flat_tensor = flat_tensor[:expected_elements]
            else:
                padding = torch.zeros(expected_elements - flat_tensor.numel(), dtype=dtype)
                flat_tensor = torch.cat([flat_tensor, padding])

        # Reshape and move to device
        return flat_tensor.reshape(shape).to(device)
    except Exception as e:
        print(f"Decompression error: {e}")
        return None

# ------------------------------
# 2. Main Functions
# ------------------------------

def run_baseline(model, tokenizer, device, prompt="Hello, how are you?"):
    """Run inference without compression for baseline"""
    print("\n----- Running baseline inference (no compression) -----")

    # Measure memory before inference
    print("Memory usage before inference:")
    measure_memory_usage()

    # Prepare input
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.input_ids,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Measure memory after inference
    print("Memory usage after inference:")
    measure_memory_usage()

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    return response

def run_with_compression(model, tokenizer, device, prompt="Hello, how are you?"):
    """Run inference with model compression"""
    print("\n----- Running inference with compression -----")

    # Step 1: Compress model weights
    print("Compressing model weights...")
    compressed_weights = []

    for i, param in enumerate(model.parameters()):
        print(f"Compressing parameter {i+1}", end="\r")
        compressed = compress_tensor(param.data)
        if compressed:
            compressed_weights.append(compressed)

            # Calculate compression ratio
            original = compressed['original_size']
            compressed_size = compressed['compressed_size']
            ratio = original / compressed_size if compressed_size > 0 else 0
            print(f"Parameter {i+1}: {original/1024:.1f} KB -> {compressed_size/1024:.1f} KB (ratio: {ratio:.1f}x)")

    print(f"Compressed {len(compressed_weights)} parameters")

    # Measure memory after compression
    print("Memory usage after compression:")
    measure_memory_usage()

    # Step 2: Empty CUDA cache and delete model to simulate reduced memory footprint
    print("Simulating reduced memory footprint...")
    del model
    torch.cuda.empty_cache()

    # Measure memory after model deletion
    print("Memory usage after model deletion:")
    measure_memory_usage()

    # Step 3: Reload minimal model structure
    print("Reloading minimal model structure...")
    model, _, _ = load_model_from_hf(MODEL_ID)

    # Step 4: Decompress weights back into model
    print("Decompressing weights...")
    params_list = list(model.parameters())

    for i, compressed_param in enumerate(compressed_weights):
        if i < len(params_list):
            print(f"Decompressing parameter {i+1}", end="\r")
            decompressed = decompress_tensor(compressed_param, device)
            if decompressed is not None:
                # Verify shapes match before assignment
                if decompressed.shape == params_list[i].shape:
                    params_list[i].data = decompressed
                else:
                    print(f"Shape mismatch for parameter {i}: expected {params_list[i].shape}, got {decompressed.shape}")

    # Measure memory after decompression
    print("Memory usage after decompression:")
    measure_memory_usage()

    # Step 5: Run inference
    print("Running inference...")
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.input_ids,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    return response

# ------------------------------
# Main execution
# ------------------------------

def main():
    # Load model and tokenizer
    model, tokenizer, device = load_model_from_hf(MODEL_ID)

    # Measure initial memory
    print("Initial memory usage:")
    measure_memory_usage()

    # Ask user which method to try
    print("\nChoose an option:")
    print("1. Run baseline inference without compression")
    print("2. Run with model compression/decompression")

    try:
        choice = input("Enter your choice (1 or 2): ")
    except:
        # Default to 1 if running in non-interactive mode
        choice = "1"

    # Run selected method
    if choice == "2":
        run_with_compression(model, tokenizer, device)
    else:
        run_baseline(model, tokenizer, device)

if __name__ == "__main__":
    main()
