# vanilla_cnn
python3 ./inference/tensorrt/export.py --onnx_model_path="./weights/vanilla_cnn/onnx_model.onnx"

# resnet18
python3 ./inference/tensorrt/export.py --onnx_model_path="./weights/resnet18/onnx_model.onnx"

# mobilenet_v2
python3 ./inference/tensorrt/export.py --onnx_model_path="./weights/mobilenet_v2/onnx_model.onnx"
