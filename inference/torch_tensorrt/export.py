import argparse
import sys
import os
import torch
import torch_tensorrt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.models import VanillaCNN, ResNet18Model, MobileNetV2Model

def main(args):
    # Model selection
    print(f"Initializing model: {args.backbone}")
    model_map = {
        "vanilla_cnn": VanillaCNN,
        "resnet18": ResNet18Model,
        "mobilenet_v2": MobileNetV2Model
    }

    # Ensure the selected backbone is valid
    if args.backbone not in model_map:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    # Load the model and weights
    model = model_map[args.backbone](num_classes=args.num_classes).cuda()
    model.load_state_dict(torch.load(args.best_model_path))
    # Apply FP16 precision if specified
    if args.precision == "fp16":
        model = model.half()
    model.eval()    
    example_input = torch.randn(1, 3, 224, 224).cuda()
    scripted_model = torch.jit.script(model)
    traced_model = torch.jit.trace(scripted_model, example_input)
    # Convert the model to Torch-TensorRT
    dtype = torch.half if args.precision=='fp16' else torch.float
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input(example_input.shape, dtype=dtype)],
        enabled_precisions={dtype},  # You can use torch.float, torch.half or torch.int8 for mixed precision
        truncate_long_and_double=True     # Handle non-TensorRT types
    )

    # Save the converted model
    torch.jit.save(trt_model, f"{os.path.dirname(args.best_model_path)}/trt_model_{args.precision}.ts")
    print("Torch-TensorRT model has been saved.")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Perform inference using a pre-trained model.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes.")
    parser.add_argument(
        "--backbone", type=str, choices=["vanilla_cnn", "resnet18", "mobilenet_v2"], 
        default="resnet18", help="Backbone model to use for export."
    )
    parser.add_argument(
        "--best_model_path", type=str, default="./weights/resnet18/best_model.pth", 
        help="Path to the saved weights of the best model."
    )
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16", 
        help="Precision for the model.")
    args = parser.parse_args()

    # Run the main function
    main(args)
