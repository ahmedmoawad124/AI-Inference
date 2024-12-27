import onnx
import onnxoptimizer

# Load the ONNX model
model = onnx.load("./weights/mobilenet_v3/onnx_model.onnx")

# Apply optimizations
optimized_model = onnxoptimizer.optimize(model)

# Save the optimized model
onnx.save(optimized_model, "./weights/mobilenet_v3/onnx_optimized_model.onnx")
