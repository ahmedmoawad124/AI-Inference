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
<table>
  <tr>
    <th>Framework / Compiler</th>
    <th>Device</th>
    <th>Precision</th>
    <th>Accuracy (%)</th>
    <th>FPS</th>
    <th>GPU Memory (MiB)</th>
    <th>GPU Utilization (%)</th>
  </tr>
  <tr>
    <td rowspan="2">PyTorch</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>87.5</td>
    <td>354.522</td>
    <td>283.00</td>
    <td>6.09</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>87.5</td>
    <td>238.779</td>
    <td>381.00</td>
    <td>10.05</td>
  </tr>
  <tr>
    <td rowspan="2">TorchScript</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>87.5</td>
    <td>353.857</td>
    <td>225.00</td>
    <td>5.60</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>87.5</td>
    <td>239.124</td>
    <td>225.00</td>
    <td>8.82</td>
  </tr>
  <tr>
    <td rowspan="2">Torch-TensorRT</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>87.5</td>
    <td>458.431</td>
    <td>195.00</td>
    <td>4.61</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>87.5</td>
    <td>726.124</td>
    <td>141.00</td>
    <td>2.84</td>
  </tr>
  <tr>
    <td rowspan="2">ONNX</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>87.5</td>
    <td>468.416</td>
    <td>357.00</td>
    <td>5.42</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>87.5</td>
    <td>535.090</td>
    <td>231.00</td>
    <td>4.41</td>
  </tr>
  <tr>
    <td rowspan="2">ONNX-TensorRT</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>87.5</td>
    <td>497.709</td>
    <td>391.00</td>
    <td>5.04</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>87.5</td>
    <td>814.335</td>
    <td>335.00</td>
    <td>3.03</td>
  </tr>
  <tr>
    <td rowspan="2">TensorRT</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>87.5</td>
    <td>501.207</td>
    <td>173.00</td>
    <td>4.35</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>87.5</td>
    <td>842.575</td>
    <td>119.00</td>
    <td>2.54</td>
  </tr>
</table>

### ResNet18
<table>
  <tr>
    <th>Framework / Compiler</th>
    <th>Device</th>
    <th>Precision</th>
    <th>Accuracy (%)</th>
    <th>FPS</th>
    <th>GPU Memory (MiB)</th>
    <th>GPU Utilization (%)</th>
  </tr>
  <tr>
    <td rowspan="2">PyTorch</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>95.67</td>
    <td>192.866</td>
    <td>395.00</td>
    <td>10.68</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>95.67</td>
    <td>76.369</td>
    <td>395.00</td>
    <td>24.18</td>
  </tr>
  <tr>
    <td rowspan="2">TorchScript</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>95.67</td>
    <td>211.668</td>
    <td>239.00</td>
    <td>9.42</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>95.67</td>
    <td>76.672</td>
    <td>239.00</td>
    <td>28.12</td>
  </tr>
  <tr>
    <td rowspan="2">Torch-TensorRT</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>95.67</td>
    <td>381.955</td>
    <td>151.00</td>
    <td>6.68</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>95.67</td>
    <td>582.762</td>
    <td>123.00</td>
    <td>4.13</td>
  </tr>
  <tr>
    <td rowspan="2">ONNX</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>95.67</td>
    <td>227.414</td>
    <td>243.00</td>
    <td>9.98</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>95.67</td>
    <td>184.349</td>
    <td>171.00</td>
    <td>11.54</td>
  </tr>
  <tr>
    <td rowspan="2">ONNX-TensorRT</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>95.67</td>
    <td>389.084</td>
    <td>347.00</td>
    <td>6.14</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>95.67</td>
    <td>620.516</td>
    <td>317.00</td>
    <td>3.93</td>
  </tr>
  <tr>
    <td rowspan="2">TensorRT</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>95.67</td>
    <td>394.255</td>
    <td>129.00</td>
    <td>6.16</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>95.67</td>
    <td>649.792</td>
    <td>101.00</td>
    <td>3.67</td>
  </tr>
</table>

### MobileNetV2
<table>
  <tr>
    <th>Framework / Compiler</th>
    <th>Device</th>
    <th>Precision</th>
    <th>Accuracy (%)</th>
    <th>FPS</th>
    <th>GPU Memory (MiB)</th>
    <th>GPU Utilization (%)</th>
  </tr>
  <tr>
    <td rowspan="2">PyTorch</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>96.31</td>
    <td>79.952</td>
    <td>419.00</td>
    <td>6.33</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>96.31</td>
    <td>78.017</td>
    <td>419.00</td>
    <td>12.40</td>
  </tr>
  <tr>
    <td rowspan="2">TorchScript</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>96.31</td>
    <td>143.242</td>
    <td>261.00</td>
    <td>7.55</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>96.31</td>
    <td>113.558</td>
    <td>263.00</td>
    <td>13.40</td>
  </tr>
  <tr>
    <td rowspan="2">Torch-TensorRT</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>96.31</td>
    <td>554.743</td>
    <td>105.00</td>
    <td>3.94</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>96.31</td>
    <td>614.645</td>
    <td>99.00</td>
    <td>3.38</td>
  </tr>
  <tr>
    <td rowspan="2">ONNX</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>96.31</td>
    <td>299.382</td>
    <td>381.00</td>
    <td>8.01</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>96.31</td>
    <td>326.883</td>
    <td>125.00</td>
    <td>8.04</td>
  </tr>
  <tr>
    <td rowspan="2">ONNX-TensorRT</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>96.31</td>
    <td>597.654</td>
    <td>303.00</td>
    <td>4.10</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>96.31</td>
    <td>665.619</td>
    <td>297.00</td>
    <td>3.59</td>
  </tr>
  <tr>
    <td rowspan="2">TensorRT</td>
    <td rowspan="2">GPU</td>
    <td>FP32</td>
    <td>96.31</td>
    <td>610.959</td>
    <td>83.00</td>
    <td>3.83</td>
  </tr>
  <tr>
    <td>FP16</td>
    <td>96.31</td>
    <td>686.453</td>
    <td>77.00</td>
    <td>3.11</td>
  </tr>
</table>

##### Important Note on ONNX Runtime vs. PyTorch:

When comparing the inference speed of ONNX Runtime with PyTorch, it's crucial to account for data transfer overhead.

PyTorch FPS calculations typically exclude the time taken to copy input data from CPU to GPU and output data back to CPU.
ONNX Runtime, on the other hand, inherently includes these data transfer times.
Therefore, a direct FPS comparison between PyTorch and ONNX Runtime can be misleading. To ensure a fair comparison, it's essential to include the data transfer times in the PyTorch FPS calculations.
