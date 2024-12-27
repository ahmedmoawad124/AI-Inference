import argparse
import sys
import os
import torch.onnx
from torchsummary import summary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.models import VanillaCNN, ResNet18Model, MobileNetV2Model

def main(args):
    """
    Main function to perform inference on a dataset using a pre-trained model.
    
    Args:
        args: Parsed command-line arguments.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available processor {}".format(device))

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
    model = model_map[args.backbone](num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.best_model_path))
    model.eval()
    print('Finished loading model!')
    summary(model, input_size=(3, 224, 224))

    # ------------------------ export -----------------------------
    output_onnx = f"{os.path.dirname(args.best_model_path)}/onnx_model.onnx"
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))

    inputs = torch.randn(1, 3, 224, 224).to(device)

    # Export the model
    torch.onnx.export(model,                     # model being run
                    inputs,                    # model input (or a tuple for multiple inputs)
                    output_onnx,               # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['logits', 'softmax_out'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'logits' : {0 : 'batch_size'},
                                    'softmax_out' : {0 : 'batch_size'}})

    print("ONNX model is Exported")
    
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
    args = parser.parse_args()

    # Run the main function
    main(args)
