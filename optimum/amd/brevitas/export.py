from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode

from optimum.exporters.onnx import onnx_export_from_model
from optimum.exporters.onnx.base import OnnxConfig
from transformers.modeling_utils import PreTrainedModel

import onnx
from onnx_tool import Model
from onnx_tool.node import create_node
from onnx_tool.tensor import Tensor

from onnx_tool.fusion import FusionPattern

from optimum.onnx.graph_transformations import check_and_save_model

import pathlib


## Pattern to find and replace with MatMulInteger
MatMul = [
    {
        'name': 'deq_linear_0',
        'op': 'DequantizeLinear',
        'attrs': [
        ],
        'inport': [],
        'outport': [[0, 'transpose_0', 0]],
    },
    {
        'name': 'transpose_0',
        'op': 'Transpose',
        'attrs': [
        ],
        'inport': [[0, 'deq_linear_0', 0]],
        'outport': [[0, 'matmul_0', 1]],
    },
    {
        'name': 'quant_linear_1',
        'op': 'DynamicQuantizeLinear',
        'attrs': [
        ],
        'inport': [],
        'outport': [[0, 'deq_linear_1', 0],
                    [1, 'deq_linear_1', 1],
                    [2, 'deq_linear_1', 2]
                    ],
    },
    {
        'name': 'deq_linear_1',
        'op': 'DequantizeLinear',
        'attrs': [
        ],
        'inport': [[0, 'quant_linear_1', 0],
                   [1, 'quant_linear_1', 1],
                   [2, 'quant_linear_1', 2],],
        'outport': [[0, 'matmul_0', 0]],
    },
    {
        'name': 'matmul_0',
        'op': 'MatMul',
        'attrs': [
        ],
        'inport': [[0, 'deq_linear_1', 0],
                   [1, 'transpose_0', 0],
                   ],
        'outport': [],
    },
]

GEMM = [
    {
        'name': 'deq_linear_0',
        'op': 'DequantizeLinear',
        'attrs': [
        ],
        'inport': [],
        'outport': [[0, 'gemm_0', 1]],
    },
    {
        'name': 'quant_linear_1',
        'op': 'DynamicQuantizeLinear',
        'attrs': [
        ],
        'inport': [],
        'outport': [[0, 'deq_linear_1', 0],
                    [1, 'deq_linear_1', 1],
                    [2, 'deq_linear_1', 2]
                    ],
    },
    {
        'name': 'deq_linear_1',
        'op': 'DequantizeLinear',
        'attrs': [
        ],
        'inport': [[0, 'quant_linear_1', 0],
                   [1, 'quant_linear_1', 1],
                   [2, 'quant_linear_1', 2],],
        'outport': [[0, 'gemm_0', 0]],
    },
    {
        'name': 'gemm_0',
        'op': 'Gemm',
        'attrs': [
        ],
        'inport': [[0, 'deq_linear_1', 0],
                   [1, 'deq_linear_0', 0],
                   ],
        'outport': [],
    },
]

def create_nodes(graph, op, name, inputs, outputs, intermediate=None, **kwargs):

    if intermediate is None:
        intermediate = []
    newnode = onnx.helper.make_node(op, inputs + intermediate, outputs, name=name, **kwargs)
    newnode = create_node(newnode)
    newnode.input = inputs + intermediate
    newnode.output = outputs
    for i in inputs:
        if i in graph.consumedby:
            graph.consumedby[i].append(name)
        if i in graph.producedby.keys():
            newnode.prevnodes.append(graph.producedby[i])
    for o in outputs:
        graph.producedby[o] = [name]
        if o in graph.consumedby.keys():
            newnode.nextnodes.append(graph.consumedby[o])
    graph.nodemap[name] = newnode
    graph.tensormap[name] = Tensor(name)

    return graph

def replace_matmul_to_matmulinteger(compute_graph, found_nodes):
    for i, found_pattern in enumerate(found_nodes):
        deq_linear = compute_graph.nodemap[found_pattern[0]]
        dyn_q = compute_graph.nodemap[found_pattern[2]]
        dq_weight = deq_linear.prevnodes[0]
        compute_graph.add_initial(f'dq_weights_0_{i}', dq_weight.value.transpose())
        compute_graph.add_initial(f'dq_weights_1_{i}', deq_linear.prevnodes[1].value)
        compute_graph.add_initial(f'dq_weights_2_{i}', deq_linear.prevnodes[2].value)


        matmul = compute_graph.nodemap[found_pattern[-1]]
        for name in found_pattern:
            if 'DynamicQuantizeLinear' in name:
                continue
            compute_graph.remove_node(name)

        compute_graph.remove_node(deq_linear.prevnodes[0].name)
        compute_graph.remove_node(deq_linear.prevnodes[1].name)
        if deq_linear.prevnodes[2].name in compute_graph.nodemap:
            compute_graph.remove_node(deq_linear.prevnodes[2].name)

        compute_graph=create_nodes(compute_graph, 'MatMulInteger', f'matmul_integer_{i}', [dyn_q.output[0], f'dq_weights_0_{i}', dyn_q.output[2], f'dq_weights_2_{i}'], [f'matmul_integer_{i}'])
        compute_graph=create_nodes(compute_graph, 'Cast', f'cast_{i}',  [f'matmul_integer_{i}'], [f'cast_{i}'], to=int(1))
        compute_graph=create_nodes(compute_graph, 'Mul', f'mulscales_{i}', [dyn_q.output[1], f'dq_weights_1_{i}'],   [f'mulscales_{i}'])
        compute_graph=create_nodes(compute_graph, 'Mul', f'mulvalues_{i}', [f'mulscales_{i}', f'cast_{i}'], [matmul.output[0]])
    return compute_graph


def replace_gemm_to_matmulinteger(compute_graph, found_nodes):
    k = 100
    for i, found_pattern in enumerate(found_nodes):
        k = i + 100
        gemm = compute_graph.nodemap[found_pattern[-1]]
        bias = gemm.input[-1]
        deq_linear = compute_graph.nodemap[found_pattern[0]]
        dyn_q = compute_graph.nodemap[found_pattern[1]]
        dq_weight = deq_linear.prevnodes[0]
        compute_graph.add_initial(f'dq_weights_0_{k}', dq_weight.value.transpose())
        compute_graph.add_initial(f'dq_weights_1_{k}', deq_linear.prevnodes[1].value)
        compute_graph.add_initial(f'dq_weights_2_{k}', deq_linear.prevnodes[2].value)
    
        matmul = compute_graph.nodemap[found_pattern[-1]]
        for name in found_pattern:
            if 'DynamicQuantizeLinear' in name:
                continue
            compute_graph.remove_node(name)
        compute_graph.remove_node(deq_linear.prevnodes[0].name)
        compute_graph.remove_node(deq_linear.prevnodes[1].name)
        if deq_linear.prevnodes[2].name in compute_graph.nodemap:
            compute_graph.remove_node(deq_linear.prevnodes[2].name)

        compute_graph=create_nodes(compute_graph, 'MatMulInteger', f'matmul_integer_{k}', [dyn_q.output[0], f'dq_weights_0_{k}', dyn_q.output[2], f'dq_weights_2_{k}'], [f'matmul_integer_{k}'])
        compute_graph=create_nodes(compute_graph, 'Cast', f'cast_{k}',  [f'matmul_integer_{k}'], [f'cast_{k}'], to=int(1))
        compute_graph=create_nodes(compute_graph, 'Mul', f'mulscales_{k}', [dyn_q.output[1], f'dq_weights_1_{k}'],   [f'mulscales_{k}'])
        compute_graph=create_nodes(compute_graph, 'Mul', f'mulvalues_{k}', [f'mulscales_{k}', f'cast_{k}'], [f'mulvalues_{k}'])
        compute_graph=create_nodes(compute_graph, 'Add', f'addbias_{k}', [bias, f'mulvalues_{k}'], [matmul.output[0]])
    return compute_graph

def find_and_insert_matmulinteger(model_path):
    print("Rewriting ONNX Graph with MatMulInteger ")

    cfg={'constant_folding':False,'node_rename':False,'if_fixed_branch':None,'fixed_topk':0,'verbose':True}
    original_output = onnx.load(model_path).graph.output
    model = Model(model_path,cfg)
    graph = model.graph

    print("Replacing MatMul with MatMulInteger")
    pattern = FusionPattern(MatMul)
    found_nodes = pattern.search_pattern(graph)
    graph = replace_matmul_to_matmulinteger(graph, found_nodes)

    print("Replacing GEMM with MatMulInteger + Add")
    pattern = FusionPattern(GEMM)
    found_nodes = pattern.search_pattern(graph)
    graph = replace_gemm_to_matmulinteger(graph, found_nodes)
    
    graph.graph_reorder_nodes()

    print("Saving the new ONNX model")
    full_path = pathlib.Path(model_path)

    graph = graph.make_graph_onnx(graph.nodemap.keys(), 'graph', graph.input, graph.output,
                                    with_initializer=True, with_shape_info=False)

    attr = {'producer_name': 'onnx_tool'}
    model_to_save = onnx.helper.make_model(graph, **attr)

    # onnx_tools might remove the output nodes from the ONNX graph, so we need to restore it.
    for out in original_output:
        if out not in model.graph.output:
            model.graph.output.append(out)

    model_to_save.ir_version = model.mproto.ir_version
    model_to_save.opset_import.pop()
    for opset in model.mproto.opset_import:
        model_to_save.opset_import.append(opset)

    check_and_save_model(model_to_save, full_path)


def onnx_export_from_quantized_model(
    quantized_model: Union["PreTrainedModel"],
    output: Union[str, Path],
    opset: Optional[int] = None,
    optimize: Optional[str] = None,
    monolith: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None,
    custom_onnx_configs: Optional[Dict[str, "OnnxConfig"]] = None,
    fn_get_submodels: Optional[Callable] = None,
    _variant: str = "default",
    preprocessors: List = None,
    device: str = "cpu",
    no_dynamic_axes: bool = False,
    task: str = "text-generation-with-past",
    use_subprocess: bool = False,
    do_constant_folding: bool = True,
    insert_matmulinteger: bool = True,
    **kwargs_shapes,
):
    with torch.no_grad(), brevitas_proxy_export_mode(quantized_model, export_manager=StdQCDQONNXManager):
        onnx_export_from_model(
            quantized_model,
            output,
            opset=opset,
            monolith=monolith,
            optimize=optimize,
            model_kwargs=model_kwargs,
            custom_onnx_configs=custom_onnx_configs,
            fn_get_submodels=fn_get_submodels,
            _variant=_variant,
            preprocessors=preprocessors,
            device=device,
            no_dynamic_axes=no_dynamic_axes,
            use_subprocess=use_subprocess,
            do_constant_folding=do_constant_folding,
            task=task,
            do_validation=False,
            no_post_process=True,
            **kwargs_shapes,
        )

    # Replace quantized GEMM and MatMul with MatMulInteger
    if insert_matmulinteger:
        find_and_insert_matmulinteger(output)
