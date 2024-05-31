import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import onnx
import torch
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from onnx_tool import Model
from onnx_tool.fusion import FusionPattern
from onnx_tool.graph import Graph
from onnx_tool.node import create_node
from onnx_tool.tensor import Tensor

from optimum.exporters.onnx import onnx_export_from_model
from optimum.exporters.onnx.base import OnnxConfig
from optimum.onnx.graph_transformations import check_and_save_model
from transformers.modeling_utils import PreTrainedModel


LOGGER = logging.getLogger(__name__)

ONNX_FLOAT32_IDENTIFIER = int(1)

## Pattern to find and replace with MatMulInteger
MATMUL = [
    {
        "name": "deq_linear_0",
        "op": "DequantizeLinear",
        "attrs": [],
        "inport": [],
        "outport": [[0, "transpose_0", 0]],
    },
    {
        "name": "transpose_0",
        "op": "Transpose",
        "attrs": [],
        "inport": [[0, "deq_linear_0", 0]],
        "outport": [[0, "matmul_0", 1]],
    },
    {
        "name": "quant_linear_1",
        "op": "DynamicQuantizeLinear",
        "attrs": [],
        "inport": [],
        "outport": [[0, "deq_linear_1", 0], [1, "deq_linear_1", 1], [2, "deq_linear_1", 2]],
    },
    {
        "name": "deq_linear_1",
        "op": "DequantizeLinear",
        "attrs": [],
        "inport": [
            [0, "quant_linear_1", 0],
            [1, "quant_linear_1", 1],
            [2, "quant_linear_1", 2],
        ],
        "outport": [[0, "matmul_0", 0]],
    },
    {
        "name": "matmul_0",
        "op": "MatMul",
        "attrs": [],
        "inport": [
            [0, "deq_linear_1", 0],
            [1, "transpose_0", 0],
        ],
        "outport": [],
    },
]

GEMM = [
    {
        "name": "deq_linear_0",
        "op": "DequantizeLinear",
        "attrs": [],
        "inport": [],
        "outport": [[0, "gemm_0", 1]],
    },
    {
        "name": "quant_linear_1",
        "op": "DynamicQuantizeLinear",
        "attrs": [],
        "inport": [],
        "outport": [[0, "deq_linear_1", 0], [1, "deq_linear_1", 1], [2, "deq_linear_1", 2]],
    },
    {
        "name": "deq_linear_1",
        "op": "DequantizeLinear",
        "attrs": [],
        "inport": [
            [0, "quant_linear_1", 0],
            [1, "quant_linear_1", 1],
            [2, "quant_linear_1", 2],
        ],
        "outport": [[0, "gemm_0", 0]],
    },
    {
        "name": "gemm_0",
        "op": "Gemm",
        "attrs": [],
        "inport": [
            [0, "deq_linear_1", 0],
            [1, "deq_linear_0", 0],
        ],
        "outport": [],
    },
]


def create_nodes(graph: Graph, op: str, name: str, inputs: List[str], outputs: List[str], **kwargs):
    newnode = onnx.helper.make_node(op, inputs, outputs, name=name, **kwargs)
    newnode = create_node(newnode)
    newnode.input = inputs
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


def replace_matmul_to_matmulinteger(graph: Graph, found_nodes: List[List[str]], node_count: int = 0):
    for found_pattern in found_nodes:
        node_count += 1

        deq_linear = graph.nodemap[found_pattern[0]]
        dyn_q = graph.nodemap[found_pattern[2]]
        dq_weight = deq_linear.prevnodes[0]
        graph.add_initial(f"dq_weights_0_{node_count}", dq_weight.value.transpose())
        graph.add_initial(f"dq_weights_1_{node_count}", deq_linear.prevnodes[1].value)
        graph.add_initial(f"dq_weights_2_{node_count}", deq_linear.prevnodes[2].value)

        matmul = graph.nodemap[found_pattern[-1]]
        for name in found_pattern:
            if "DynamicQuantizeLinear" in name:
                continue
            graph.remove_node(name)

        graph.remove_node(deq_linear.prevnodes[0].name)
        if deq_linear.prevnodes[1].name in graph.nodemap:
            graph.remove_node(deq_linear.prevnodes[1].name)
        if deq_linear.prevnodes[2].name in graph.nodemap:
            graph.remove_node(deq_linear.prevnodes[2].name)

        graph = create_nodes(
            graph,
            "MatMulInteger",
            f"matmul_integer_{node_count}",
            [dyn_q.output[0], f"dq_weights_0_{node_count}", dyn_q.output[2], f"dq_weights_2_{node_count}"],
            [f"matmul_integer_{node_count}"],
        )
        graph = create_nodes(
            graph,
            "Cast",
            f"cast_{node_count}",
            [f"matmul_integer_{node_count}"],
            [f"cast_{node_count}"],
            to=ONNX_FLOAT32_IDENTIFIER,
        )
        graph = create_nodes(
            graph,
            "Mul",
            f"mulscales_{node_count}",
            [dyn_q.output[1], f"dq_weights_1_{node_count}"],
            [f"mulscales_{node_count}"],
        )
        graph = create_nodes(
            graph,
            "Mul",
            f"mulvalues_{node_count}",
            [f"mulscales_{node_count}", f"cast_{node_count}"],
            [matmul.output[0]],
        )
    return graph


def replace_gemm_to_matmulinteger(graph: Graph, found_nodes: List[List[str]], node_count: int = 0):
    for found_pattern in found_nodes:
        node_count += 1

        gemm = graph.nodemap[found_pattern[-1]]
        bias = gemm.input[-1]
        deq_linear = graph.nodemap[found_pattern[0]]
        dyn_q = graph.nodemap[found_pattern[1]]
        dq_weight = deq_linear.prevnodes[0]
        graph.add_initial(f"dq_weights_0_{node_count}", dq_weight.value.transpose())
        graph.add_initial(f"dq_weights_1_{node_count}", deq_linear.prevnodes[1].value)
        graph.add_initial(f"dq_weights_2_{node_count}", deq_linear.prevnodes[2].value)

        matmul = graph.nodemap[found_pattern[-1]]
        for name in found_pattern:
            if "DynamicQuantizeLinear" in name:
                continue
            graph.remove_node(name)
        graph.remove_node(deq_linear.prevnodes[0].name)
        if deq_linear.prevnodes[1].name in graph.nodemap:
            graph.remove_node(deq_linear.prevnodes[1].name)
        if deq_linear.prevnodes[2].name in graph.nodemap:
            graph.remove_node(deq_linear.prevnodes[2].name)

        graph = create_nodes(
            graph,
            "MatMulInteger",
            f"matmul_integer_{node_count}",
            [dyn_q.output[0], f"dq_weights_0_{node_count}", dyn_q.output[2], f"dq_weights_2_{node_count}"],
            [f"matmul_integer_{node_count}"],
        )
        graph = create_nodes(
            graph,
            "Cast",
            f"cast_{node_count}",
            [f"matmul_integer_{node_count}"],
            [f"cast_{node_count}"],
            to=ONNX_FLOAT32_IDENTIFIER,
        )
        graph = create_nodes(
            graph,
            "Mul",
            f"mulscales_{node_count}",
            [dyn_q.output[1], f"dq_weights_1_{node_count}"],
            [f"mulscales_{node_count}"],
        )
        graph = create_nodes(
            graph,
            "Mul",
            f"mulvalues_{node_count}",
            [f"mulscales_{node_count}", f"cast_{node_count}"],
            [f"mulvalues_{node_count}"],
        )
        graph = create_nodes(
            graph, "Add", f"addbias_{node_count}", [bias, f"mulvalues_{node_count}"], [matmul.output[0]]
        )
    return graph


def find_and_insert_matmulinteger(model_path: str):
    # onnx_tool requires python 3.9+
    if sys.version_info[0] == 3 and sys.version_info[1] <= 8:
        raise RuntimeError("onnx_tool requires Python 3.9 or higher")

    LOGGER.info("Rewriting ONNX Graph with MatMulInteger")
    model_path = os.path.join(model_path, "model.onnx")
    cfg = {"constant_folding": False, "node_rename": False, "if_fixed_branch": None, "fixed_topk": 0, "verbose": False}

    onnx_model = onnx.load(model_path)

    # Extract model output
    original_output = copy.deepcopy(onnx_model.graph.output)

    model = Model(onnx_model, cfg)
    graph = model.graph

    pattern = FusionPattern(MATMUL)
    found_matmul_nodes = pattern.search_pattern(graph)
    matmul_node_count = len(found_matmul_nodes)
    LOGGER.info(f"Replacing {matmul_node_count} MatMul nodes with MatMulInteger")
    graph = replace_matmul_to_matmulinteger(graph, found_matmul_nodes)

    pattern = FusionPattern(GEMM)
    found_gemm_nodes = pattern.search_pattern(graph)
    gemm_node_count = len(found_gemm_nodes)
    LOGGER.info(f"Replacing {gemm_node_count} Gemm nodes with MatMulInteger + Add")
    graph = replace_gemm_to_matmulinteger(graph, found_gemm_nodes, matmul_node_count)

    graph.graph_reorder_nodes()

    LOGGER.info("Saving the new ONNX model")
    full_path = Path(model_path)

    graph = graph.make_graph_onnx(
        graph.nodemap.keys(), "graph", graph.input, graph.output, with_initializer=True, with_shape_info=False
    )

    attr = {"producer_name": "onnx_tool"}
    model_to_save = onnx.helper.make_model(graph, **attr)

    # onnx_tools might remove the output nodes from the ONNX graph, so we need to restore it.
    for out in original_output:
        if out not in model_to_save.graph.output:
            model_to_save.graph.output.append(out)
    from pdb import set_trace

    set_trace()
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
