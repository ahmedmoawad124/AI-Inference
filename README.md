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

## Models Training
#### VanillaCNN: ```python3 train.py --backbone="vanilla_cnn" --saved_model_dir="./weights/vanilla_cnn/"```
#### ResNet18: ```python3 train.py --backbone="resnet18" --saved_model_dir="./weights/resnet18/"```
#### MobileNetV2: ```python3 train.py --backbone="mobilenet_v2" --saved_model_dir="./weights/mobilenet_v2/"```

## Weights Downloading
```bash ./weights/download.sh```

## Models Exporting
#### Torch-Script: ```bash ./inference/torch_script/export.sh```
#### Torch-Tensorrt: ```bash ./inference/torch_tensorrt/export.sh```
#### ONNX: ```bash ./inference/onnx/export.sh```

## Models Benchmarking
#### Torch-Script: ```python3 ./inference/torch_script/benchmark.py```
#### Torch-Tensorrt: ```python3 ./inference/torch_tensorrt/benchmark.py```
#### ONNX: ```python3 ./inference/onnx/benchmark.py```
