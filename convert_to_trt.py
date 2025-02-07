import tensorrt as trt
import sys
import os

# Initialize TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, max_workspace_size=1 << 30, enable_fp16=True, enable_int8=False):
    """
    Builds a TensorRT engine from an ONNX model.

    Args:
        onnx_file_path (str): Path to the input ONNX model.
        engine_file_path (str): Path to save the output TensorRT engine.
        max_workspace_size (int, optional): Maximum GPU memory (in bytes) the builder can use. Defaults to 1GB.
        enable_fp16 (bool, optional): Whether to enable FP16 precision. Defaults to True.
        enable_int8 (bool, optional): Whether to enable INT8 precision. Defaults to False.
    
    Returns:
        trt.ICudaEngine: The built TensorRT engine, or None if failed.
    """
    try:
        # Create TensorRT builder, network, and parser
        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(
                 flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
             ) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser, \
             builder.create_builder_config() as config:
            
            # Set the maximum GPU memory that the builder can use at runtime
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
            
            # (Optional) Enable FP16 mode for faster inference if supported
            if enable_fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("FP16 mode enabled.")
            else:
                print("FP16 mode not enabled or not supported.")
            
            # (Optional) Enable INT8 mode for even faster inference with quantization
            if enable_int8:
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    print("INT8 mode enabled.")
                    # Note: INT8 calibration is required for accurate results
                    # You need to provide a calibrator here
                    # Example: config.int8_calibrator = MyCalibrator()
                    print("INT8 mode requires a calibrator. Please implement and provide one.")
                else:
                    print("INT8 mode not supported on this platform.")
            
            # Parse the ONNX model
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for idx in range(parser.num_errors):
                        print(f"Parser Error {idx}: {parser.get_error(idx)}")
                    return None
            
            # (Optional) Define an optimization profile if your model has dynamic input shapes
            # Uncomment and modify the following lines if needed
            # profile = builder.create_optimization_profile()
            # input_tensor = network.get_input(0)
            # profile.set_shape(input_tensor.name, (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
            # config.add_optimization_profile(profile)
            
            # Build the serialized engine
            print("Building the TensorRT engine. This may take a while...")
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                print("ERROR: Failed to create the serialized TensorRT engine.")
                return None
            
            # Save the serialized engine to file
            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)
            print(f"TensorRT engine successfully saved to '{engine_file_path}'")
            return serialized_engine

    except trt.Error as e:
        print(f"TensorRT Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine.")
    parser.add_argument("onnx_model", help="Path to the input ONNX model.")
    parser.add_argument("trt_engine", help="Path to save the output TensorRT engine.")
    parser.add_argument("--max_workspace_size", type=int, default=1 << 30, help="Maximum GPU memory for TensorRT (in bytes). Default is 1GB.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision.")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 precision (requires calibrator).")
    
    args = parser.parse_args()

    onnx_model = args.onnx_model
    trt_engine = args.trt_engine
    max_workspace_size = args.max_workspace_size
    enable_fp16 = args.fp16
    enable_int8 = args.int8

    # Check if ONNX model exists
    if not os.path.isfile(onnx_model):
        print(f"ERROR: The ONNX model file '{onnx_model}' does not exist.")
        sys.exit(1)
    
    # Optionally, check if the output directory exists
    output_dir = os.path.dirname(trt_engine)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory '{output_dir}'.")
        except OSError as e:
            print(f"ERROR: Could not create output directory '{output_dir}'. {e}")
            sys.exit(1)
    
    engine = build_engine(
        onnx_model,
        trt_engine,
        max_workspace_size=max_workspace_size,
        enable_fp16=enable_fp16,
        enable_int8=enable_int8
    )
    if engine:
        print("Engine building completed successfully.")
    else:
        print("Engine building failed.")

if __name__ == "__main__":
    main()
