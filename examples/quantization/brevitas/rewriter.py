import onnx_graphsurgeon as gs
import onnx
import numpy as np

# Here we'll register a function to do all the subgraph-replacement heavy-lifting.
# NOTE: Since registered functions are entirely reusable, it may be a good idea to
# refactor them into a separate module so you can use them across all your models.
@gs.Graph.register()
def replace_with_matmulinteger(self, inputs, intermediate, outputs, count=0):
    # Disconnect output nodes of all input tensors
    for inp in inputs.values():
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()
    bias = inputs.get('bias', None)
    # Insert the new node.
    matmulint = self.layer(op='MatMulInteger', inputs=[inputs['inp_float'], intermediate['int_weight'], inputs['inp_zp'], intermediate['weight_zp']], outputs=[f'matmul_{count}'])
    cast = self.layer(op='Cast', inputs=matmulint, outputs=[f'cast_{count}'], attrs= {'to': int(1)})
    mul_scale = self.layer(op='Mul', inputs=[inputs['inp_scale'], intermediate['weight_scale']], outputs=[f'mul_scale_{count}'])
    if bias is not None:
        mul_matmulin = self.layer(op='Mul', inputs=[mul_scale[0], cast[0]], outputs=[f'mul_matmulin{count}'])
        bias_node = self.layer(op='Add', inputs=[mul_matmulin[0], bias], outputs = outputs)
        last_node = bias_node
    else:
        mul_matmulin = self.layer(op='Mul', inputs=[mul_scale[0], cast[0]], outputs=outputs)
        last_node = mul_matmulin

    return last_node

def quantize(value, scale, zero_point):
    return (np.round(value/scale)+ zero_point).astype(np.int8)

def find_node(starting, destination):
    if hasattr(starting, 'op') and starting.op == destination:
        return starting
    elif len(starting.inputs) > 0:
        return find_node(starting.inputs[0], destination)
    else:
        return None

def extract_values(node):
    while len(node.inputs) > 0:
        node = node.inputs[0]
    if hasattr(node, 'values'):
        return node.values
    elif hasattr(node, 'attrs'):
        return node.attrs['value'].values

def extract_inp_intermediate_out_matmul(gemm_node):
    input = gemm_node.inputs[0]
    weights = gemm_node.inputs[1]
    dynamic_quant = find_node(input, 'DynamicQuantizeLinear')
    if dynamic_quant is None:
        return None, None, None
    # TODO: Find a way to support both DQ and QCDQ
    weight_quant = find_node(weights, 'DequantizeLinear')

    inp_float = dynamic_quant.outputs[0]
    inp_scale = dynamic_quant.outputs[1]
    inp_zp = dynamic_quant.outputs[2]
    weight_inp = extract_values(weight_quant.inputs[0])
    weight_scale =  extract_values(weight_quant.inputs[1])
    weight_zp =  extract_values(weight_quant.inputs[2])
    int_weight = quantize(weight_inp.transpose(), weight_scale, weight_zp)
    weight_out = gemm_node.outputs[0]
    inps = {'inp_float': inp_float,
            'inp_scale': inp_scale,
            'inp_zp': inp_zp}
    intermediate = {'int_weight': int_weight,
                    'weight_scale': weight_scale, 
                    'weight_zp': weight_zp}
    return inps, intermediate, [weight_out]

def extract_inp_intermediate_out_gemm(gemm_node):
    inps, intermediate, weight_out = extract_inp_intermediate_out_matmul(gemm_node)
    if inps is None:
        return None, None, None
    bias = gemm_node.inputs[2]
    inps['bias'] = bias
    return inps, intermediate, weight_out

def rewrite_graph(model_path):
    graph = gs.import_onnx(onnx.load(model_path))

    gemm = [x for x in graph.nodes if x.op == 'Gemm']
    matmul = [x for x in graph.nodes if x.op == 'MatMul']
    for m in matmul:
        inp, interm, out = extract_inp_intermediate_out_matmul(m)
        if inp is None:
            continue
        graph.replace_with_matmulinteger(inp, interm, out)
    for m in gemm:
        inp, interm, out = extract_inp_intermediate_out_gemm(m)
        if inp is None:
            continue
        graph.replace_with_matmulinteger(inp, interm, out)

    # Remove the now-dangling subgraph.
    graph.cleanup().toposort()

    # TODO: We need a check to know if our model has external data
    # If that's not possible, we can treat all our models as if they need to be saved on external data
    onnx.save(gs.export_onnx(graph), model_path)

