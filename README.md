# <p align="center">Mastering AI Inference</p>
<p align="center">
  <img src="https://community.intel.com/t5/image/serverpage/image-id/23222i737060699A80CB3E/image-size/large?v=v2&px=999" alt="PyTorch Logo" width="125">
  <img src="https://www.edureka.co/blog/wp-content/uploads/2017/06/hadoop-logo-1.png" alt="PyTorch Logo" width="106">
  <img src="https://d2mk45aasx86xg.cloudfront.net/Top_Python_libraries_for_Machine_Learning_29078075a6.webp" alt="PyTorch Logo" width="108">
  <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2020/04/tensorrt-logo.png" alt="TensorRT Logo" width="79">
  <img src="https://developer.nvidia.com/sites/default/files/akamai/onnx.png" alt="TensorRT Logo" width="98">
  <img src="https://neousys-web-bucket.s3.us-west-1.amazonaws.com/img/market/intel-openvivo-toolkit-300.gif" alt="TensorRT Logo" width="126">
  <img src="https://raw.githubusercontent.com/apache/tvm-site/main/images/logo/tvm-logo-small.png" alt="TensorRT Logo" width="125">
</p>

## Conda environment and installing the packages:
Create the environment by running:

```conda env create -f conda_env.yml```

if you add new packages, you can update the conda environment with:

```conda env update -f conda_env.yml```

To activate it:

```conda activate ai_inference_env```

## Dataset Downloading
```python3 ./datasets/download_dataset.py```

## Run Train script
#### VanillaCNN: 
```python3 train.py --backbone="vanilla_cnn" --saved_model_dir="./weights/vanilla_cnn/"```
#### ResNet18: 
```python3 train.py --backbone="resnet18" --saved_model_dir="./weights/resnet18/"```
#### MobileNetV2: 
```python3 train.py --backbone="mobilenet_v2" --saved_model_dir="./weights/mobilenet_v2/"```

## Run Weights Download script
```bash ./weights/download.sh```

## Run Models Export scripts
#### Torch-Script: 
```bash ./inference/torch_script/export.sh```
#### Torch-Tensorrt: 
```bash ./inference/torch_tensorrt/export.sh```
#### ONNX: 
```bash ./inference/onnx/export.sh```

## Run Benchmark scripts
#### PyTorch: 
```python3 ./inference/pytorch/benchmark.py```
#### Torch-Script: 
```python3 ./inference/torch_script/benchmark.py```
#### Torch-Tensorrt: 
```python3 ./inference/torch_tensorrt/benchmark.py```
#### ONNX: 
```python3 ./inference/onnx/benchmark.py```

## Benchmark Results
The benchmark results were conducted on the following setup:

- **GPU**: NVIDIA Quadro T2000 (4 GB)
- **CUDA Version**: 11.8

### VanillaCNN
| Framework / Compiler | Device  | Precision | Accuracy (%) | FPS      | GPU Memory (MiB)  | GPU Utilization (%)  |
|----------------------|---------|-----------|--------------|----------|-------------------|----------------------|
| PyTorch              | GPU     | FP32      | 87.5         | 355.853  | 295.00            | 5.78                 |
| TorchScript          | GPU     | FP32      | 87.5         | 358.822  | 237.00            | 5.74                 |
| Torch-TensorRT       | GPU     | FP32      | 87.5         | 419.277  | 183.00            | 4.17                 |
| ONNX                 | GPU     | FP32      | 87.5         | 449.473  | 357.00            | 4.62                 |
| ONNX-TensorRT        | GPU     | FP32      | 87.5         | 478.035  | 389.00            | 4.79                 |

### ResNet18
| Framework / Compiler | Device  | Precision | Accuracy (%) | FPS      | GPU Memory (MiB)  | GPU Utilization (%)  |
|----------------------|---------|-----------|--------------|----------|-------------------|----------------------|
| PyTorch              | GPU     | FP32      | 95.67        | 183.542  | 309.00            | 9.52                 |
| TorchScript          | GPU     | FP32      | 95.67        | 212.470  | 251.00            | 9.10                 |
| Torch-TensorRT       | GPU     | FP32      | 95.67        | 357.560  | 145.00            | 5.78                 |
| ONNX                 | GPU     | FP32      | 95.67        | 243.092  | 243.00            | 9.50                 |
| ONNX-TensorRT        | GPU     | FP32      | 95.67        | 389.137  | 343.00            | 5.92                 |

### MobileNetV2
| Framework / Compiler | Device  | Precision | Accuracy (%) | FPS      | GPU Memory (MiB)  | GPU Utilization (%)  |
|----------------------|---------|-----------|--------------|----------|-------------------|----------------------|
| PyTorch              | GPU     | FP32      | 96.31        | 102.647  | 331.00            | 7.59                 |
| TorchScript          | GPU     | FP32      | 96.31        | 120.876  | 273.00            | 6.74                 |
| Torch-TensorRT       | GPU     | FP32      | 96.31        | 506.939  | 93.00             | 3.85                 |
| ONNX                 | GPU     | FP32      | 96.31        | 273.726  | 377.00            | 7.99                 |
| ONNX-TensorRT        | GPU     | FP32      | 96.31        | 581.654  | 297.00            | 3.80                 |

###### Important Note on ONNX Runtime vs. PyTorch:

When comparing the inference speed of ONNX Runtime with PyTorch, it's crucial to account for data transfer overhead.

PyTorch FPS calculations typically exclude the time taken to copy input data from CPU to GPU and output data back to CPU.
ONNX Runtime, on the other hand, inherently includes these data transfer times.
Therefore, a direct FPS comparison between PyTorch and ONNX Runtime can be misleading. To ensure a fair comparison, it's essential to include the data transfer times in the PyTorch FPS calculations.
