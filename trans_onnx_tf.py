import onnx

# https://github.com/sithu31296/PyTorch-ONNX-TFLite
# Load the ONNX model
model = onnx.load('head2-copy.onnx')

# Check that the IR is well formed
print(onnx.checker.check_model(model))

# Print a Human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

# python3 converter.py onnx2tnn /home/zhou/Documents/python/DualStyleGAN/head2-copy.onnx
