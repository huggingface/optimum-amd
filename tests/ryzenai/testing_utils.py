# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.


RYZEN_PRETRAINED_MODEL_IMAGE_CLASSIFICATION = [
    "amd/efficientnet-es",
    "amd/ese_vovnet39b",
    "amd/inception_v4",
    "amd/mnasnet_b1",
    "amd/mobilenet_v2_1.0_224",
    "amd/resnet50",
    "amd/squeezenet",
]


RYZEN_PRETRAINED_MODEL_OBJECT_DETECTION = [
    "amd/retinaface",
    "amd/yolov3",
    "amd/yolov5s",
    "amd/yolov8m",
    "amd/yolox-s",
]

RYZEN_PRETRAINED_MODEL_IMAGE_SEGMENTATION = [
    "amd/HRNet",
    "amd/SemanticFPN",
]

RYZEN_PRETRAINED_MODEL_IMAGE_TO_IMAGE = ["amd/PAN", "amd/rcan", "amd/sesr"]

RYZEN_PRETRAINED_MODEL_CUSTOM_TASKS = ["amd/movenet"]

PYTORCH_TIMM_MODEL = {
    "default-timm-config": {
        "timm/inception_v3.tf_adv_in1k": ["image-classification"],
        "timm/tf_efficientnet_b0.in1k": ["image-classification"],
        "timm/resnetv2_50x1_bit.goog_distilled_in1k": ["image-classification"],
        "timm/cspdarknet53.ra_in1k": ["image-classification"],
        "timm/cspresnet50.ra_in1k": ["image-classification"],
        "timm/cspresnext50.ra_in1k": ["image-classification"],
        "timm/densenet121.ra_in1k": ["image-classification"],
        "timm/dla102.in1k": ["image-classification"],
        "timm/dpn107.mx_in1k": ["image-classification"],
        "timm/ecaresnet101d.miil_in1k": ["image-classification"],
        "timm/efficientnet_b1_pruned.in1k": ["image-classification"],
        "timm/inception_resnet_v2.tf_ens_adv_in1k": ["image-classification"],
        "timm/fbnetc_100.rmsp_in1k": ["image-classification"],
        "timm/xception41.tf_in1k": ["image-classification"],
        "timm/senet154.gluon_in1k": ["image-classification"],
        "timm/seresnext26d_32x4d.bt_in1k": ["image-classification"],
        "timm/hrnet_w18.ms_aug_in1k": ["image-classification"],
        "timm/inception_v3.gluon_in1k": ["image-classification"],
        "timm/inception_v4.tf_in1k": ["image-classification"],
        "timm/mixnet_s.ft_in1k": ["image-classification"],
        "timm/mnasnet_100.rmsp_in1k": ["image-classification"],
        "timm/mobilenetv2_100.ra_in1k": ["image-classification"],
        "timm/mobilenetv3_small_050.lamb_in1k": ["image-classification"],
        "timm/nasnetalarge.tf_in1k": ["image-classification"],
        "timm/tf_efficientnet_b0.ns_jft_in1k": ["image-classification"],
        "timm/pnasnet5large.tf_in1k": ["image-classification"],
        "timm/regnetx_002.pycls_in1k": ["image-classification"],
        "timm/regnety_002.pycls_in1k": ["image-classification"],
        "timm/res2net101_26w_4s.in1k": ["image-classification"],
        "timm/res2next50.in1k": ["image-classification"],
        "timm/resnest101e.in1k": ["image-classification"],
        "timm/spnasnet_100.rmsp_in1k": ["image-classification"],
        "timm/resnet18.fb_swsl_ig1b_ft_in1k": ["image-classification"],
        "timm/wide_resnet101_2.tv_in1k": ["image-classification"],
        "timm/tresnet_l.miil_in1k": ["image-classification"],
    }
}
