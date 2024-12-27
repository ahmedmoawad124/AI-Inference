# vanilla_cnn
python3 ./inference/torch_script/export.py --backbone="vanilla_cnn" --best_model_path="./weights/vanilla_cnn/best_model.pth"

# resnet18
python3 ./inference/torch_script/export.py --backbone="resnet18" --best_model_path="./weights/resnet18/best_model.pth"

# mobilenet_v2
python3 ./inference/torch_script/export.py --backbone="mobilenet_v2" --best_model_path="./weights/mobilenet_v2/best_model.pth"
