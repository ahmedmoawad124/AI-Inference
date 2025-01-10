import argparse
import os
import numpy as np
import cv2
import onnxruntime
from imutils import paths
from tqdm import tqdm
import time


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

    # Load the model
    if args.provider == "cuda":
        providers = [("CUDAExecutionProvider", {'cudnn_conv_use_max_workspace':'1', 'cudnn_conv_algo_search': 'EXHAUSTIVE'})]
    elif args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.provider == "tensorrt":
        providers = [("TensorrtExecutionProvider", {"trt_engine_cache_enable": True,  
                                                    "trt_engine_cache_path": f"{os.path.dirname(args.onnx_model_path)}/trt_cache"
                                                    }), ("CUDAExecutionProvider", {'cudnn_conv_use_max_workspace':'1', 
                                                                                   'cudnn_conv_algo_search': 'EXHAUSTIVE'})]
    
    # Set up session options for optimization
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # Enable all optimizations

    ort_session = onnxruntime.InferenceSession(args.onnx_model_path, sess_options=session_options, providers=providers)


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
        ort_inputs = {ort_session.get_inputs()[0].name: preprocessed_image}
    
        # Make predictions
        start_time = time.time()
        logits, softmax_out = ort_session.run(None, ort_inputs)
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
        "--provider", type=str, choices=["cuda", "cpu", "tensorrt"], default="cuda")
    parser.add_argument(
        "--onnx_model_path", type=str, default="./weights/resnet18/onnx_model_fp16.onnx", 
        help="Path to the saved onnx model."
    )
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16", 
        help="Precision for the model.")
    args = parser.parse_args()

    # Run the main function
    main(args)
