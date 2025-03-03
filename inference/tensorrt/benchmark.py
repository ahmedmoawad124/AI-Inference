import argparse
import numpy as np
from tqdm import tqdm
import time
import GPUtil
import trt_utils


def benchmark_model(model_path, num_warmup_iterations, num_iterations, precision):
    """
    Benchmark a TRT Engine.
    
    Args:
        model_path (str): Path to the TRT Engine.
        num_warmup_iterations (int): Number of warmup iterations.
        num_iterations (int): Number of inference iterations.
    
    Returns:
        float: Average inference time (in milliseconds).
        float: Maximum GPU memory used (in MB).
        float: Maximum GPU utilization (in percentage).
    """

    model = trt_utils.TRTInference(model_path)

    # Prepare the input data
    dtype = np.float16 if precision=='fp16' else np.float32
    input_data = np.random.randn(1, 3, 224, 224).astype(dtype)

    # Warmup the model
    for _ in range(num_warmup_iterations):
        output = model(input_data)

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
        start_time = time.time()
        
        output = model(input_data)
        logits, softmax_out = output['logits'], output['softmax_out']
        
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)

    average_inference_time = np.mean(inference_times)
    average_gpu_memory = np.mean(gpu_memory_list)
    average_gpu_utilization = np.mean(gpu_utilization_list)
    return average_inference_time, average_gpu_memory, average_gpu_utilization


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Benchmark TRT Engine on TensorRT.")
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
    precisions = ["fp32", "fp16"]
    # Benchmark models
    for model_name in models:
        print(f"Benchmarking {model_name} model:")
        for precision in precisions:
            print(f"    Benchmarking for {precision} precision:")
            avg_time, average_gpu_mem, average_gpu_util = benchmark_model(
                    f"./weights/{model_name}/trt_engine_{precision}_Quadro_T2000_sm75.engine", 
                    args.num_warmup_iterations, args.num_iterations, precision
                )
            print(f"      Average inference time: {avg_time:.2f} ms")
            print(f"      Average GPU memory used: {average_gpu_mem:.2f} MB")
            print(f"      Average GPU utilization: {average_gpu_util:.2f} %")
            print(f"      FPS: {1000 / avg_time:.3f}")

if __name__ == "__main__":
    main()
