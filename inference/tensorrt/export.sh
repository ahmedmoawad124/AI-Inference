# vanilla_cnn
python3 ./inference/tensorrt/export.py --onnx_model_path="./weights/vanilla_cnn/onnx_model_fp32.onnx" --precision="fp32"
python3 ./inference/tensorrt/export.py --onnx_model_path="./weights/vanilla_cnn/onnx_model_fp16.onnx" --precision="fp16"

# resnet18
python3 ./inference/tensorrt/export.py --onnx_model_path="./weights/resnet18/onnx_model_fp32.onnx" --precision="fp32"
python3 ./inference/tensorrt/export.py --onnx_model_path="./weights/resnet18/onnx_model_fp16.onnx" --precision="fp16"

# mobilenet_v2
python3 ./inference/tensorrt/export.py --onnx_model_path="./weights/mobilenet_v2/onnx_model_fp32.onnx" --precision="fp32"
python3 ./inference/tensorrt/export.py --onnx_model_path="./weights/mobilenet_v2/onnx_model_fp16.onnx" --precision="fp16"
