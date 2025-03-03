import argparse
import sys
import os
import numpy as np
import cv2
import torch
from imutils import paths
from tqdm import tqdm
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.models import VanillaCNN, ResNet18Model, MobileNetV2Model


def preprocess_image(image, precision):
    """
    Preprocess an image for inference.
    
    Args:
        image (numpy.ndarray): Input image to preprocess.
    
    Returns:
        numpy.ndarray: Preprocessed image ready for inference.
    """
    # Resize the image to the required dimensions
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    dtype = "float16" if precision=="fp16" else "float32"
    img = img.astype(dtype) / 255.0  # Normalize pixel values to [0, 1]
    
    # Reorder dimensions to "channels first" and add batch dimension
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    
    return img


def main(args):
    """
    Main function to perform inference on a dataset using a pre-trained model.
    
    Args:
        args: Parsed command-line arguments.
    """
    # Set device for computation (GPU or CPU)
    device = args.device
    if device == "cuda":
        assert torch.cuda.is_available()
        seed = 0
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Using device: {device}")

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
    model = model_map[args.backbone](num_classes=len(args.classes_list)).to(device)
    model.load_state_dict(torch.load(args.best_model_path))
    if args.precision == "fp16":
        model.half()
    model.eval()  # Set the model to evaluation mode
        
    # Create a mapping of class names to one-hot vectors
    class_to_one_hot = {
        cls: np.eye(len(args.classes_list))[i] for i, cls in enumerate(args.classes_list)
    }

    # Perform inference
    correct_predictions = 0
    image_paths = sorted(list(paths.list_images(args.data_path)))
    infernce_time = 0
    print("Starting inference on dataset...")
    for image_path in tqdm(image_paths, desc="Processing images"):
        # Load and preprocess the image
        image = cv2.imread(image_path)
        preprocessed_image = preprocess_image(image, args.precision)

        # Make predictions
        start_time = time.time()  # Including data transfers between the CPU and GPU (and vice versa) ensures a fair comparison with ONNX Runtime.
        input_tensor = torch.from_numpy(preprocessed_image).to(device)
        logits, softmax_out = model(input_tensor)
        logits = logits.detach().cpu().numpy()
        softmax_out = softmax_out.detach().cpu().numpy()
        infernce_time += (time.time() - start_time)
        predicted_class = softmax_out.argmax()

        # Get the ground-truth class label from the image path
        class_name = os.path.basename(os.path.dirname(image_path))
        target_class = np.argmax(class_to_one_hot[class_name])

        # Update the count of correct predictions
        if predicted_class == target_class:
            correct_predictions += 1

    # Calculate and print accuracy
    total_images = len(image_paths)
    accuracy = (correct_predictions / total_images) * 100
    fps = total_images / infernce_time
    print(f"Accuracy: {correct_predictions}/{total_images} ({accuracy:.2f}%)")
    print(f"Total time: {infernce_time:.2f} seconds")
    print(f"FPS: {fps:.3f}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Perform inference using a pre-trained model.")
    parser.add_argument(
        "--data_path", type=str, default="../chest_xray/test/", 
        help="Path to the dataset containing images for inference."
    )
    parser.add_argument(
        "--classes_list", type=list, default=["NORMAL", "PNEUMONIA"], 
        help="List of class names in the correct order."
    )
    parser.add_argument(
        "--backbone", type=str, choices=["vanilla_cnn", "resnet18", "mobilenet_v2"], 
        default="resnet18", help="Backbone model to use for inference."
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--best_model_path", type=str, default="./weights/resnet18/best_model.pth", 
        help="Path to the saved weights of the best model."
    )
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16", 
        help="Precision for the model.")
    args = parser.parse_args()

    # Run the main function
    main(args)
