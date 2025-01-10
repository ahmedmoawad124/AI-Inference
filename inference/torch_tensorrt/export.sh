# vanilla_cnn
python3 ./inference/torch_tensorrt/export.py --backbone="vanilla_cnn" --best_model_path="./weights/vanilla_cnn/best_model.pth" --precision="fp32"
python3 ./inference/torch_tensorrt/export.py --backbone="vanilla_cnn" --best_model_path="./weights/vanilla_cnn/best_model.pth" --precision="fp16"

# resnet18
python3 ./inference/torch_tensorrt/export.py --backbone="resnet18" --best_model_path="./weights/resnet18/best_model.pth" --precision="fp32"
python3 ./inference/torch_tensorrt/export.py --backbone="resnet18" --best_model_path="./weights/resnet18/best_model.pth" --precision="fp16"

# mobilenet_v2
python3 ./inference/torch_tensorrt/export.py --backbone="mobilenet_v2" --best_model_path="./weights/mobilenet_v2/best_model.pth" --precision="fp32"
python3 ./inference/torch_tensorrt/export.py --backbone="mobilenet_v2" --best_model_path="./weights/mobilenet_v2/best_model.pth" --precision="fp16"
