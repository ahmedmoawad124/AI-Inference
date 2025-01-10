import os
import collections
from pathlib import Path
import logging
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initialize CUDA context

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Define the log format
    handlers=[logging.StreamHandler()]  # Output logs to the console
)

IS_TRT10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
                

def get_available_gpu_memory():
    """Returns the available GPU memory in bytes."""
    free, total = cuda.mem_get_info()
    return free
    
    
def export_engine(onnx_file_path, fp16_mode=False, dynamic_batch_size=False):
    # Setup and checks
    LOGGER = logging.getLogger()
    prefix = "TensorRT:"
    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    assert Path(onnx_file_path).exists(), f"failed to export ONNX file: {onnx_file_path}"
    device = cuda.Device(0)  # Select the first GPU (device index 0)
    gpu_name = device.name().replace(" ","_")     # GPU name
    compute_capability = device.compute_capability()  # GPU architecture (Compute Capability)
    precisions = 'fp16' if fp16_mode else 'fp32'
    engine_file = os.path.join(os.path.dirname(onnx_file_path), f"trt_engine_{precisions}_{gpu_name}_sm{compute_capability[0]}{compute_capability[1]}.engine")
    logger = trt.Logger(trt.Logger.INFO)

    # Engine builder
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Set max workspace size to available GPU memory
    workspace = get_available_gpu_memory()

    if IS_TRT10 and workspace > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    elif workspace > 0:  # TensorRT versions 7, 8
        config.max_workspace_size = workspace
    
    LOGGER.info(f"\n{prefix} Setting max workspace size to {workspace / (1 << 30):.2f} GB")
    
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    half = builder.platform_has_fast_fp16 and fp16_mode

    # Read ONNX file
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_file_path):
        raise RuntimeError(f"failed to load ONNX file: {onnx_file_path}")
    
    # Set dynamic batch size
    if dynamic_batch_size:
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_shape = input_tensor.shape
            
            if input_shape[0] == -1:  # Batch size is dynamic
                # Set dynamic batch size range: (min=1, opt=8, max=32)
                profile.set_shape(
                    input_tensor.name,
                    (1, *input_shape[1:]),  # Minimum shape
                    (8, *input_shape[1:]),  # Optimal shape
                    (32, *input_shape[1:])  # Maximum shape
                )
            else:
                LOGGER.warning(f"Input {input_tensor.name} does not have a dynamic batch size")
        
        # Add the profile to the config
        config.add_optimization_profile(profile)
    
    else:
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            shape = list(input_tensor.shape)
            shape[0] = 1  # Fix batch size to 1
            network.get_input(i).shape = tuple(shape)
            
    # Network inputs
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    LOGGER.info(f"{prefix} building {'FP' + ('16' if half else '32')} engine as {engine_file}")
    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    # Write file
    build = builder.build_serialized_network if IS_TRT10 else builder.build_engine
    with build(network, config) as engine, open(engine_file, "wb") as t:
        # Model
        t.write(engine if IS_TRT10 else engine.serialize())

    return engine_file


class TRTInference(object):
    def __init__(self, engine_path, processing_batch_size=1):
        self.logger = trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)        
        # Model context
        try:
            self.context = self.engine.create_execution_context()
        except Exception as e:  # model is None
            print(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
            raise e
        self.processing_batch_size = processing_batch_size
        self.input_names = []
        self.output_names = []
        self.bindings = self.get_bindings()
        self.bindings_addr = collections.OrderedDict((n, v.ptr) for n, v in self.bindings.items())

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_bindings(self) -> collections.OrderedDict:
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = collections.OrderedDict()
        
        bindings_num = range(self.engine.num_io_tensors) if IS_TRT10 else range(self.engine.num_bindings)
        for i in bindings_num:
            if IS_TRT10:
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if is_input:
                    if -1 in tuple(self.engine.get_tensor_shape(name)):
                        assert self.processing_batch_size <= self.engine.get_tensor_profile_shape(name, 0)[2][0], "processing_batch_size > Max batch size"
                        shape = self.engine.get_tensor_shape(i)
                        shape[0] = self.processing_batch_size
                        self.context.set_input_shape(i, shape)
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
                shape = tuple(self.engine.get_tensor_shape(name))
            else:  # TensorRT < 10.0
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                if self.engine.binding_is_input(i):
                    if -1 in tuple(self.engine.get_binding_shape(i)):  # dynamic
                        assert self.processing_batch_size <= self.engine.get_profile_shape(0, i)[2][0], "processing_batch_size > Max batch size"
                        shape = self.engine.get_binding_shape(i)
                        shape[0] = self.processing_batch_size
                        self.context.set_binding_shape(i, shape)
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
                shape = tuple(self.context.get_binding_shape(i))
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).cuda()
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())
            
        return bindings


    def __call__(self, *inputs):
        assert len(inputs) == len(self.input_names), "number of inputs not equal to expected number"

        self.bindings_addr.update({n: inputs[i].contiguous().data_ptr() for i, n in enumerate(self.input_names)})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs
