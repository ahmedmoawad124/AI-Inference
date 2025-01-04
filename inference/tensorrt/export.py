import argparse
import trt_utils

def main(args):
    trt_utils.export_engine(args.onnx_model_path, dynamic_batch_size=False)

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="tensorrt engine exporter")
    parser.add_argument(
        "--onnx_model_path", type=str, default="./weights/mobilenet_v2/onnx_model.onnx", 
        help="Path to the saved weights of the onnx model."
    )
    args = parser.parse_args()

    # Run the main function
    main(args)
