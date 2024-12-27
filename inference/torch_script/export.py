import argparse
import sys
import os
import torch
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
    model.eval()

    # Convert the model to Torch-Script
    ts_model = torch.jit.script(model)

    # Save the converted model
    torch.jit.save(ts_model, f"{os.path.dirname(args.best_model_path)}/ts_model.ts")
    print("Torch-Script model has been saved as 'ts_model.ts'.")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Perform inference using a pre-trained model.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes.")
    parser.add_argument(
        "--backbone", type=str, choices=["vanilla_cnn", "resnet18", "mobilenet_v2"], 
        default="vanilla_cnn", help="Backbone model to use for export."
    )
    parser.add_argument(
        "--best_model_path", type=str, default="./weights/vanilla_cnn/best_model.pth", 
        help="Path to the saved weights of the best model."
    )
    args = parser.parse_args()

    # Run the main function
    main(args)
