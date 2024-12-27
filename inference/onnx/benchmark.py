import argparse
import os
import numpy as np
import cv2
import onnxruntime
from tqdm import tqdm
import time
import GPUtil
import psutil


def benchmark_model(model_path, provider, num_warmup_iterations, num_iterations):
    """
    Benchmark an ONNX model on a given provider.
    
    Args:
        model_path (str): Path to the ONNX model.
        provider (str): Provider to use for inference ("cuda", "cpu", "tensorrt").
        num_warmup_iterations (int): Number of warmup iterations.
        num_iterations (int): Number of inference iterations.
    
    Returns:
        float: Average inference time (in milliseconds).
        float: Maximum GPU memory used (in MB).
        float: Maximum GPU utilization (in percentage).
    """
    # Set up session options for optimization
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # Enable all optimizations

    # Create the ONNX session
    if provider == "cuda":
        providers = [("CUDAExecutionProvider", {'cudnn_conv_use_max_workspace':'1', 'cudnn_conv_algo_search': 'EXHAUSTIVE'})]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider == "tensorrt":
        providers = [("TensorrtExecutionProvider", {"trt_engine_cache_enable": True,  
                                                    "trt_engine_cache_path": f"{os.path.dirname(model_path)}/trt_cache"
                                                    }), ("CUDAExecutionProvider", {'cudnn_conv_use_max_workspace':'1', 
                                                                                   'cudnn_conv_algo_search': 'EXHAUSTIVE'})]
    
    ort_session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)

    # Prepare the input data
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}

    # Warmup the model
    for _ in range(num_warmup_iterations):
        ort_session.run(None, ort_inputs)

    # Measure performance
    inference_times = []
    gpu_memory_list = []
    gpu_utilization_list = []
    cpu_utilization_list = []
    
    for _ in tqdm(range(num_iterations), desc="    Processing images"):
        # Get GPU memory and utilization
        gpu_memory = GPUtil.getGPUs()[0].memoryUsed
        gpu_utilization = GPUtil.getGPUs()[0].load * 100
        cpu_utilization = psutil.cpu_percent(interval=None)
        gpu_memory_list.append(gpu_memory)
        gpu_utilization_list.append(gpu_utilization)
        cpu_utilization_list.append(cpu_utilization)

        # Perform inference
        start_time = time.time()
        ort_session.run(None, ort_inputs)
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)

    average_inference_time = np.mean(inference_times)
    average_gpu_memory = np.mean(gpu_memory_list)
    average_gpu_utilization = np.mean(gpu_utilization_list)
    average_cpu_utilization = np.mean(cpu_utilization_list)
    return average_inference_time, average_gpu_memory, average_gpu_utilization, average_cpu_utilization


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
    providers = ["cuda", "cpu", "tensorrt"]

    # Benchmark models
    for model_name in models:
        print(f"Benchmarking {model_name} model:")
        for provider in providers:
            print(f"  Benchmarking on {provider} provider:")
            avg_time, average_gpu_mem, average_gpu_util, average_cpu_util = benchmark_model(
                    f"./weights/{model_name}/onnx_model.onnx", provider, 
                    args.num_warmup_iterations, args.num_iterations
                )
            print(f"      Average inference time: {avg_time:.2f} ms")
            print(f"      Average GPU memory used: {average_gpu_mem:.2f} MB")
            print(f"      Average GPU utilization: {average_gpu_util:.2f} %")
            print(f"      Average CPU utilization: {average_cpu_util:.2f} %")
            print(f"      FPS: {1000 / avg_time:.3f}")

if __name__ == "__main__":
    main()