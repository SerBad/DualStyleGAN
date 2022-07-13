import onnx
import onnxruntime

# https://github.com/sithu31296/PyTorch-ONNX-TFLite
# Load the ONNX model
path = './head2-copy.onnx'
path = './head2-copy2.onnx'
model = onnx.load(path)

# Check that the IR is well formed
print(onnx.checker.check_model(model))

session = onnxruntime.InferenceSession(path)
print(session.get_inputs())

# Print a Human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

# python3 converter.py onnx2tnn /home/zhou/Documents/python/DualStyleGAN/head2-copy.onnx
# python3.10 /home/zhou/Documents/python/TNN/tools/onnx2tnn/onnx-converter/onnx2tnn.py /home/zhou/Documents/python/DualStyleGAN/head2-copy.onnx -version=v1.0 -optimize=1 -half=0 -o ./output_dir/ -input_shape input:1,3,1024,1024