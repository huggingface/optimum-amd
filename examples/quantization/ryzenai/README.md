# Quantization
Optimum-amd provides a tool that enables you to apply quantization on many models hosted on the Hugging Face Hub using our RyzenAIOnnxQuantizer.

## Static quantization
The quantization process is abstracted via the AutoQuantizationConfig and the RyzenAIOnnxQuantizer classes. The former allows you to specify how quantization should be done, while the latter effectively handles quantization.

You can read the [VAI_Q_ONNX user guide](https://gitenterprise.xilinx.com/VitisAI-CP/vai_q_onnx/blob/dev/README.md) to learn about VAI_Q_ONNX quantization.

### Creating an AutoQuantizationConfig
The AutoQuantizationConfig class is used to specify how quantization should be done. The class can be initialized using the ipu_cnn_config() method.
```python
from optimum.amd.ryzenai import AutoQuantizationConfig
quantization_config = AutoQuantizationConfig.ipu_cnn_config()

```

### Creating an RyzenAIOnnxQuantizer
The RyzenAIOnnxQuantizer class is used to quantize your ONNX model. The class can be initialized using the from_pretrained() method.
Using a local ONNX model from a directory.
```python
from optimum.amd.ryzenai import RyzenAIOnnxQuantizer
quantizer = RyzenAIOnnxQuantizer.from_pretrained("path/to/model")
```

### Running the Example
The RyzenAIOnnxQuantizer class can be used to quantize statically your ONNX model. When applying post-training static quantization, we need to generate the calibration dataset in order to perform the calibration step.

To quantize a image classification model such as "timm/resnet50.a1_in1k", use:

```bash
python quantize_image_classification_model.py --model-id timm/resnet50.a1_in1k
```

For all the options, please check:

```bash
python quantize_image_classification_model.py --help
```