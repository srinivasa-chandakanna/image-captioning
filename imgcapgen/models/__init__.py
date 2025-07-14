# imgcapgen/models/__init__.py
"""
Allows direct import of different EncoderDecoder architectures from imgcapgen.models.

Example usage:
    from imgcapgen.models import (
        ScratchCNN_LSTM,
        ResNet_LSTM,
        ResNetFineTune_LSTM,
        ResNetFineTune2_LSTM,
        ResNetFineTune2_Attention_LSTM
    )
"""

from .scratchCNN_LSTM import EncoderDecoder as ScratchCNN_LSTM
from .resnet_LSTM import EncoderDecoder as ResNet_LSTM
from .resnet_finetune_LSTM import EncoderDecoder as ResNetFineTune_LSTM
from .resnet_finetune2_LSTM import EncoderDecoder as ResNetFineTune2_LSTM
from .resnet_finetune2_attention_LSTM import EncoderDecoder as ResNetFineTune2_Attention_LSTM

__all__ = [
    "ScratchCNN_LSTM",
    "ResNet_LSTM",
    "ResNetFineTune_LSTM",
    "ResNetFineTune2_LSTM",
    "ResNetFineTune2_Attention_LSTM"
]
