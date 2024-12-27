import argparse
import numpy as np
import torch
import torch_tensorrt
from tqdm import tqdm
import time
import GPUtil


def benchmark_model(model_path, num_warmup_iterations, num_iterations):
    """
    Benchmark an ONNX model on a given provider.
    
    Args:
        model_path (str): Path to the ONNX model.
        num_warmup_iterations (int): Number of warmup iterations.
        num_iterations (int): Number of inference iterations.
    
    Returns:
        float: Average inference time (in milliseconds).
        float: Maximum GPU memory used (in MB).
        float: Maximum GPU utilization (in percentage).
    """

    model = torch.jit.load(model_path).cuda()
    model.eval()  # Set the model to evaluation mode

    # Prepare the input data
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup the model
    for _ in range(num_warmup_iterations):
        input_tensor = torch.from_numpy(input_data).cuda()
        logits, softmax_out = model(input_tensor)
        logits = logits.detach().cpu().numpy()
        softmax_out = softmax_out.detach().cpu().numpy()

    # Measure performance
    inference_times = []
    gpu_memory_list = []
    gpu_utilization_list = []
    
    for _ in tqdm(range(num_iterations), desc="    Processing images"):
        # Get GPU memory and utilization
        gpu_memory = GPUtil.getGPUs()[0].memoryUsed
        gpu_utilization = GPUtil.getGPUs()[0].load * 100
        gpu_memory_list.append(gpu_memory)
        gpu_utilization_list.append(gpu_utilization)

        # Perform inference
        start_time = time.time() # Including data transfers between the CPU and GPU (and vice versa) ensures a fair comparison with ONNX Runtime.
        
        input_tensor = torch.from_numpy(input_data).cuda()
        logits, softmax_out = model(input_tensor)
        logits = logits.detach().cpu().numpy()
        softmax_out = softmax_out.detach().cpu().numpy()
        
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)

    average_inference_time = np.mean(inference_times)
    average_gpu_memory = np.mean(gpu_memory_list)
    average_gpu_utilization = np.mean(gpu_utilization_list)
    return average_inference_time, average_gpu_memory, average_gpu_utilization


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Benchmark pre-trained models on different providers.")
    parser.add_argument(
        "--num_warmup_iterations", type=int, default=30, 
        help="Number of warmup iterations."
    )
    parser.add_argument(
        "--num_iterations", type=int, default=1500, 
        help="Number of inference iterations."
    )
    args = parser.parse_args()

    # Models and providers to benchmark
    models = ["vanilla_cnn", "resnet18", "mobilenet_v2"]

    # Benchmark models
    for model_name in models:
        print(f"Benchmarking {model_name} model:")
        avg_time, average_gpu_mem, average_gpu_util = benchmark_model(
                f"./weights/{model_name}/trt_model.ts", 
                args.num_warmup_iterations, args.num_iterations
            )
        print(f"      Average inference time: {avg_time:.2f} ms")
        print(f"      Average GPU memory used: {average_gpu_mem:.2f} MB")
        print(f"      Average GPU utilization: {average_gpu_util:.2f} %")
        print(f"      FPS: {1000 / avg_time:.3f}")

if __name__ == "__main__":
    main()