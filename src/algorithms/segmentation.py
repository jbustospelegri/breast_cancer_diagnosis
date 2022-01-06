from typing import Union, io

import segmentation_models as sm

from tensorflow.keras.layers import BatchNormalization

from algorithms.classification import GeneralModel
from utils.config import SEGMENTATION_METRICS, SEGMENTATION_LOSS, IMG_SHAPE


class UnetGeneralModel(GeneralModel):

    __name__ = 'UnetGeneral'
    loss = SEGMENTATION_LOSS
    metrics = list(SEGMENTATION_METRICS.values())
    shape: tuple = IMG_SHAPE

    def __init__(self, backbone: str, framework: str = 'tf.keras', weights: str = 'imagenet'):

        sm.set_framework(framework)
        super(UnetGeneralModel, self).__init__(
            baseline=sm.Unet(backbone, encoder_weights=weights, encoder_freeze=True),
            preprocess_func=sm.get_preprocessing(backbone)
        )
        self.backbone = backbone

    def create_model(self, fc_top: str = None):
        self.model = self.baseline

    def set_trainable_layers(self, unfrozen_layers: str):
        super(UnetGeneralModel, self).set_trainable_layers(unfrozen_layers=unfrozen_layers)

        for layer in self.baseline.layers:
            # A parte de las capas a descongelar del modelo que forman parte del encoder, se descongelan
            # las capas decoder y las de batch normalization
            if 'decoder_' in layer.name or 'center_' in layer.name or isinstance(layer, BatchNormalization):
                layer.trainable = True

        self.baseline.get_layer('final_conv').trainable = True

    def get_preprocessing_func(self):
        return sm.get_preprocessing(self.backbone)


class UnetVGG16Model(UnetGeneralModel):

    __name__ = 'UnetVGG16'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': ['block5_conv1', 'block5_conv2', 'block5_conv3'],
        '2FT': ['block4_conv1', 'block4_conv2', 'block4_conv3'],
        '3FT': ['block3_conv1', 'block3_conv2', 'block3_conv3'],
        '4FT': ['block2_conv1', 'block2_conv2']
    }

    def __init__(self, weights: Union[str, io] = None):
        super(UnetVGG16Model, self).__init__(backbone='vgg16', weights=weights)


class UnetResnet50Model(UnetGeneralModel):

    __name__ = 'UnetResnet50'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': [
            'stage4_unit1_conv1', 'stage4_unit1_conv2', 'stage4_unit1_conv3', 'stage4_unit2_conv1',
            'stage4_unit2_conv2', 'stage4_unit2_conv3', 'stage4_unit3_conv1', 'stage4_unit3_conv2',
            'stage4_unit3_conv3'
        ],
        '2FT': [
            'stage3_unit1_conv1', 'stage3_unit1_conv2', 'stage3_unit1_conv3', 'stage3_unit2_conv1',
            'stage3_unit2_conv2', 'stage3_unit2_conv3', 'stage3_unit3_conv1', 'stage3_unit3_conv2',
            'stage3_unit3_conv3', 'stage3_unit4_conv1', 'stage3_unit4_conv2', 'stage3_unit4_conv3',
            'stage3_unit5_conv1', 'stage3_unit5_conv2', 'stage3_unit5_conv3', 'stage3_unit6_conv1',
            'stage3_unit6_conv2', 'stage3_unit6_conv3'
        ],
        '3FT': [
            'stage2_unit1_conv1', 'stage2_unit1_conv2', 'stage2_unit1_conv3', 'stage2_unit2_conv1',
            'stage2_unit2_conv2', 'stage2_unit2_conv3', 'stage2_unit3_conv1', 'stage2_unit3_conv2',
            'stage2_unit3_conv3', 'stage2_unit4_conv1', 'stage2_unit4_conv2', 'stage2_unit4_conv3'
        ],
        '4FT': [
            'stage1_unit1_conv1', 'stage1_unit1_conv2', 'stage1_unit1_conv3', 'stage1_unit2_conv1',
            'stage1_unit2_conv2', 'stage1_unit2_conv3', 'stage1_unit3_conv1', 'stage1_unit3_conv2',
            'stage1_unit3_conv3'
        ]
    }

    def __init__(self, weights: Union[str, io] = None):
        super(UnetResnet50Model, self).__init__(backbone='resnet50', weights=weights)


class UnetInceptionV3Model(UnetGeneralModel):

    __name__ = 'UnetInceptionV3'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': [
            'conv2d_89', 'conv2d_90', 'conv2d_91', 'conv2d_92', 'conv2d_86', 'conv2d_87', 'conv2d_88', 'conv2d_93',
            'conv2d_85'
        ],
        '2FT': [
            'conv2d_80', 'conv2d_81', 'conv2d_82', 'conv2d_83', 'conv2d_77', 'conv2d_78', 'conv2d_79', 'conv2d_84',
            'conv2d_76',
        ],
        '3FT': [
            'conv2d_72', 'conv2d_73', 'conv2d_74', 'conv2d_75', 'conv2d_70', 'conv2d_71'
        ],
        '4FT': [
            'conv2d_64', 'conv2d_65', 'conv2d_66', 'conv2d_67', 'conv2d_68', 'conv2d_61', 'conv2d_62', 'conv2d_63',
            'conv2d_69', 'conv2d_60',
        ]
    }

    def __init__(self, weights: Union[str, io] = None):
        super(UnetInceptionV3Model, self).__init__(backbone='inceptionv3', weights=weights)


class UnetDenseNetModel(UnetGeneralModel):
    __name__ = 'UnetDenseNet'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': [
            'conv5_block1_1_conv', 'conv5_block1_2_conv', 'conv5_block2_1_conv', 'conv5_block2_2_conv',
            'conv5_block3_1_conv', 'conv5_block3_2_conv', 'conv5_block4_1_conv', 'conv5_block4_2_conv',
            'conv5_block5_1_conv', 'conv5_block5_2_conv', 'conv5_block6_1_conv', 'conv5_block6_2_conv',
            'conv5_block7_1_conv', 'conv5_block7_2_conv', 'conv5_block8_1_conv', 'conv5_block8_2_conv',
            'conv5_block9_1_conv', 'conv5_block9_2_conv', 'conv5_block10_1_conv', 'conv5_block10_2_conv',
            'conv5_block11_1_conv', 'conv5_block11_2_conv', 'conv5_block12_1_conv', 'conv5_block12_2_conv',
            'conv5_block13_1_conv', 'conv5_block13_2_conv', 'conv5_block14_1_conv', 'conv5_block14_2_conv',
            'conv5_block15_1_conv', 'conv5_block15_2_conv', 'conv5_block16_1_conv', 'conv5_block16_2_conv'
        ],
        '2FT': [
            'conv4_block1_1_conv', 'conv4_block1_2_conv', 'conv4_block2_1_conv', 'conv4_block2_2_conv',
            'conv4_block3_1_conv', 'conv4_block3_2_conv', 'conv4_block4_1_conv', 'conv4_block4_2_conv',
            'conv4_block5_1_conv', 'conv4_block5_2_conv', 'conv4_block6_1_conv', 'conv4_block6_2_conv',
            'conv4_block7_1_conv', 'conv4_block7_2_conv', 'conv4_block8_1_conv', 'conv4_block8_2_conv',
            'conv4_block9_1_conv', 'conv4_block9_2_conv', 'conv4_block10_1_conv', 'conv4_block10_2_conv',
            'conv4_block11_1_conv', 'conv4_block11_2_conv', 'conv4_block12_1_conv', 'conv4_block12_2_conv',
            'conv4_block13_1_conv', 'conv4_block13_2_conv', 'conv4_block14_1_conv', 'conv4_block14_2_conv',
            'conv4_block15_1_conv', 'conv4_block15_2_conv', 'conv4_block16_1_conv', 'conv4_block16_2_conv',
            'conv4_block17_1_conv', 'conv4_block17_2_conv', 'conv4_block18_1_conv', 'conv4_block18_2_conv',
            'conv4_block19_1_conv', 'conv4_block19_2_conv', 'conv4_block20_1_conv', 'conv4_block20_2_conv',
            'conv4_block21_1_conv', 'conv4_block21_2_conv', 'conv4_block22_1_conv', 'conv4_block22_2_conv',
            'conv4_block23_1_conv', 'conv4_block23_2_conv', 'conv4_block24_1_conv', 'conv4_block24_2_conv'
        ],
        '3FT': [
            'conv3_block1_1_conv', 'conv3_block1_2_conv', 'conv3_block2_1_conv', 'conv3_block2_2_conv',
            'conv3_block3_1_conv', 'conv3_block3_2_conv', 'conv3_block4_1_conv', 'conv3_block4_2_conv',
            'conv3_block5_1_conv', 'conv3_block5_2_conv', 'conv3_block6_1_conv', 'conv3_block6_2_conv',
            'conv3_block7_1_conv', 'conv3_block7_2_conv', 'conv3_block8_1_conv', 'conv3_block8_2_conv',
            'conv3_block9_1_conv', 'conv3_block9_2_conv', 'conv3_block10_1_conv', 'conv3_block10_2_conv',
            'conv3_block11_1_conv', 'conv3_block11_2_conv', 'conv3_block12_1_conv', 'conv3_block12_2_conv'
        ],
        '4FT': [
            'conv2_block1_1_conv', 'conv2_block1_2_conv', 'conv2_block2_1_conv', 'conv2_block2_2_conv',
            'conv2_block3_1_conv', 'conv2_block3_2_conv', 'conv2_block4_1_conv', 'conv2_block4_2_conv',
            'conv2_block5_1_conv', 'conv2_block5_2_conv', 'conv2_block6_1_conv', 'conv2_block6_2_conv'
        ]
    }

    def __init__(self, weights: Union[str, io] = None):
        super(UnetDenseNetModel, self).__init__(backbone='densenet121', weights=weights)
