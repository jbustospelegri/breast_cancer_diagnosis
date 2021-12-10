import numpy as np

from typing import Callable, io, Union, Tuple
from time import process_time

from segmentation_models.models.unet import Unet
from segmentation_models.metrics import iou_score
from tensorflow.keras import Model, Sequential, optimizers, callbacks
from tensorflow.keras.backend import count_params
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.applications import resnet50, densenet, vgg16, inception_v3
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.preprocessing.image import DataFrameIterator
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy

from src.utils.functions import get_path, get_number_of_neurons


class GeneralModel:

    __name__ = 'GeneralModel'
    model: Model = None
    callbakcs = {}
    loss = CategoricalCrossentropy()
    metrics = ['accuracy']
    history: History = None
    shape = (224, 224, 3)
    LAYERS_DICT = {
        '0FT': [],
        '1FT': ['block3_maxpool1', 'block3_maxpool2', 'block3_conv1'],
        '2FT': ['block2_maxpool', 'block2_conv1'],
        '3FT': ['block1_maxpool', 'block1_conv1'],
        '4FT': []
    }

    def __init__(self, n: int, baseline: Model = None, preprocess_func: Callable = None):
        self.baseline = baseline if baseline is not None else self.create_baseline()
        self.n = n
        self.preprocess_func = preprocess_func
        self.create_model()

    def create_baseline(self):
        """
        Función que permite crear una estructura básica compuesta por un conjunto de capas convolucionales. Este metodo
        será sobreescrito por las clases heredadas.
        """
        baseline = Sequential()

        baseline.add(Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation='relu', name='block1_conv1'))
        baseline.add(MaxPooling2D(pool_size=(2, 2), name='block1_maxpool'))

        baseline.add(Conv2D(16, (5, 5), strides=(1, 1), padding="valid", activation='relu', name='block2_conv1'))
        baseline.add(MaxPooling2D(pool_size=(2, 2), name='block2_maxpool'))

        baseline.add(Conv2D(14, (3, 3), strides=(1, 1), padding="same", activation='relu', name='block3_conv1'))
        baseline.add(MaxPooling2D(pool_size=(2, 2), name='block3_maxpool1'))
        baseline.add(MaxPooling2D(pool_size=(2, 2), name='block3_maxpool2'))

        return baseline

    def create_model(self):
        """
        Función utilizada para crear la estructura de un modelo. Esta, estará formada por la estructura básica devuelta
        por self.baseline juntamente con dos capas FC de 512 neuronas con función de activación relu. La capa de salida
        estará compuesta por una capa de salida FC con tantas neuronas como clases existan en el dataset y con función
        de activación softmax
        """

        input = Input(shape=self.shape)
        x = self.baseline(input, training=False)
        x = GlobalAveragePooling2D()(x)

        # neurons = get_number_of_neurons(x.get_shape().as_list())
        # while neurons > 15:
        #     x = Dense(neurons, activation='relu', kernel_constraint=maxnorm(3))(x)
        #     x = Dropout(0.2)(x)
        #     neurons = get_number_of_neurons(x.get_shape().as_list())
        x = Dense(128, activation='relu', kernel_constraint=maxnorm(3))(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu', kernel_constraint=maxnorm(3))(x)
        x = Dropout(0.2)(x)

        output = Dense(self.n, activation='softmax')(x)

        self.model = Model(inputs=input, outputs=output)

    def set_trainable_layers(self, unfrozen_layers: str):
        """
        Función utilizada para setear layers del modelo como entrenables. Este metodo será sobreescrito por las clases
        heredadas
        """
        self.baseline.trainable = False

        if unfrozen_layers == 'ALL':
            self.baseline.trainable = True

        elif unfrozen_layers in self.LAYERS_DICT.keys():
            list_keys = sorted(self.LAYERS_DICT.keys(), key=lambda x: int(x[0]))
            train_layers = \
                [l for layers in list_keys[:list_keys.index(unfrozen_layers) + 1] for l in self.LAYERS_DICT[layers]]
            for layer in train_layers:
                self.baseline.get_layer(layer).trainable = True

        else:
            raise ValueError(f'Unfrozen layers parametrization for {unfrozen_layers}')

    def start_train(self, train_data: DataFrameIterator, val_data: DataFrameIterator, epochs: int, batch_size: int,
                    opt: optimizers = Adam(1e-3), unfrozen_layers: str = 'ALL') -> Tuple[float, int]:
        """
        Función que compila el modelo, añade los callbacks definidos por el usuario y entrena el modelo
        :param train_data: dataframe iterator con los datos de train
        :param val_data: dataframe iterator con los datos de validación
        :param epochs: número de épocas con las que realizar el entrenamiento
        :param batch_size: tamaño del batch
        :param opt: algorítmo de optimización de gradiente descendente

        """

        # Se configura si los layers serán entrenables o no
        self.set_trainable_layers(unfrozen_layers=unfrozen_layers)

        # Se compila el modelo
        self.model.compile(optimizer=opt, loss=self.loss, metrics=self.metrics)

        start = process_time()
        self.history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            verbose=2,
            callbacks=list(self.callbakcs.values()),
            steps_per_epoch=train_data.samples // batch_size,
            validation_steps=val_data.samples // batch_size,
        )
        return process_time() - start, len(self.history.history['loss'])

    def register_metric(self, *args: Union[Callable, str]):
        for arg in args:
            self.metrics.append(arg)

    def register_callback(self, **kargs: callbacks):
        self.callbakcs = {**self.callbakcs, **kargs}

    def train_from_scratch(self, train_data, val_data, epochs: int, batch_size: int, opt: optimizers = None):
        """
            Función utilizada para entrenar completamente el modelo.
        """
        t, e = self.start_train(train_data, val_data, epochs, batch_size, opt, unfrozen_layers='ALL')
        return t, e

    def extract_features(self, train_data, val_data, epochs: int, batch_size: int, opt: optimizers = None):
        """
        Función utilizada para aplicar un proceso de extract features de modo que se conjela la arquitectura definida en
        self.baseline y se entrenan las últimas capas de la arquitectura
        """
        t, e = self.start_train(train_data, val_data, epochs, batch_size, opt, unfrozen_layers='0FT')
        return t, e

    def fine_tunning(self, train_data, val_data, epochs: int, batch_size: int, opt: optimizers = None,
                     unfrozen_layers: str = '1FT'):
        """
        Función utilizada para aplicar un proceso de transfer learning de modo que se conjelan n - k capas. Las k capas
        entrenables en la arquitectura definida por self.baseline se determinarán a partir del método
        set_trainable_layers
        """
        t, e = self.start_train(train_data, val_data, epochs, batch_size, opt, unfrozen_layers=unfrozen_layers)
        return t, e

    def save_model(self, dirname: str, model_name: str):
        """
        Función utilizada para almacenar el modelo entrenado
        :param dirname: directorio en el cual se almacenara el modelo
        :param model_name: nombre del archivo para almacenar el modelo
        """
        self.model.save(get_path(dirname, model_name))

    def predict(self, *args, **kwargs):
        """
        Función utilizada para generar las predicciones de un conjunto de datos dado
        """
        return self.model.predict(*args, **kwargs)

    def get_trainable_layers(self) -> int:
        return len([l for l in self.baseline.layers if l.trainable])

    def get_trainable_params(self) -> int:
        return int(np.sum([count_params(p) for p in self.baseline.trainable_weights]))


class VGG16Model(GeneralModel):

    __name__ = 'VGG16'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool'],
        '2FT': ['block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool'],
        '3FT': ['block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool'],
        '4FT': ['block2_conv1', 'block2_conv2', 'block2_pool']
    }
    shape = (224, 224, 3)

    def __init__(self, n: int, weights: Union[str, io] = None):
        super(VGG16Model, self).__init__(
            n=n, baseline=vgg16.VGG16(include_top=False, weights=weights, input_shape=self.shape),
            preprocess_func=vgg16.preprocess_input
        )


class Resnet50Model(GeneralModel):

    __name__ = 'ResNet50'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': [
            'conv5_block3_1_conv', 'conv5_block3_1_bn', 'conv5_block3_1_relu', 'conv5_block3_2_conv',
            'conv5_block3_2_bn', 'conv5_block3_2_relu', 'conv5_block3_3_conv', 'conv5_block3_3_bn', 'conv5_block3_add',
            'conv5_block3_out'
        ],
        '2FT': [
            'conv5_block2_1_conv', 'conv5_block2_1_bn', 'conv5_block2_1_relu', 'conv5_block2_2_conv',
            'conv5_block2_2_bn', 'conv5_block2_2_relu', 'conv5_block2_3_conv', 'conv5_block2_3_bn', 'conv5_block2_add',
            'conv5_block2_out'
        ],
        '3FT': [
            'conv5_block1_1_conv', 'conv5_block1_1_bn', 'conv5_block1_1_relu', 'conv5_block1_2_conv',
            'conv5_block1_2_bn', 'conv5_block1_2_relu', 'conv5_block1_0_conv', 'conv5_block1_3_conv',
            'conv5_block1_0_bn', 'conv5_block1_3_bn', 'conv5_block1_add', 'conv5_block1_out'
        ],
        '4FT': [
            'conv4_block6_1_conv', 'conv4_block6_1_bn', 'conv4_block6_1_relu', 'conv4_block6_2_conv',
            'conv4_block6_2_bn', 'conv4_block6_2_relu', 'conv4_block6_3_conv', 'conv4_block6_3_bn', 'conv4_block6_add',
            'conv4_block6_out'
        ]
    }
    shape = (224, 224, 3)

    def __init__(self, n: int, weights: Union[str, io] = None):
        super(Resnet50Model, self).__init__(
            n=n, baseline=resnet50.ResNet50(include_top=False, weights=weights, input_shape=self.shape),
            preprocess_func=resnet50.preprocess_input
        )


class InceptionV3Model(GeneralModel):

    __name__ = 'InceptionV3'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': [
            'conv2d_89', 'batch_normalization_89', 'activation_89', 'conv2d_90', 'batch_normalization_90',
            'activation_90', 'conv2d_91', 'batch_normalization_91', 'activation_91', 'concatenate_1',
            'conv2d_92', 'batch_normalization_92', 'activation_92', 'conv2d_86', 'batch_normalization_86',
            'activation_86', 'conv2d_87', 'batch_normalization_87',  'activation_87', 'mixed9_1', 'conv2d_88',
            'batch_normalization_88', 'activation_88', 'average_pooling2d_8', 'conv2d_93', 'batch_normalization_93',
            'activation_93', 'conv2d_85', 'batch_normalization_85', 'activation_85', 'mixed10'
        ],
        '2FT': [
            'conv2d_80', 'batch_normalization_80', 'activation_80', 'conv2d_81', 'batch_normalization_81',
            'activation_81', 'conv2d_82', 'batch_normalization_82', 'activation_82', 'concatenate',
            'conv2d_83', 'batch_normalization_83', 'activation_83', 'conv2d_77', 'batch_normalization_77',
            'activation_77', 'conv2d_78', 'batch_normalization_78',  'activation_78', 'mixed9_0', 'conv2d_79',
            'batch_normalization_79', 'activation_79', 'average_pooling2d_7', 'conv2d_84', 'batch_normalization_84',
            'activation_84', 'conv2d_76', 'batch_normalization_76', 'activation_76', 'mixed9'
        ],
        '3FT': [
            'conv2d_72', 'batch_normalization_72', 'activation_72', 'conv2d_73', 'batch_normalization_73',
            'activation_73', 'conv2d_74', 'batch_normalization_74', 'activation_74', 'conv2d_75',
            'batch_normalization_75', 'activation_75', 'conv2d_70', 'batch_normalization_70', 'activation_70',
            'conv2d_71', 'batch_normalization_71', 'activation_71', 'max_pooling2d_3', 'mixed8'
        ],
        '4FT': [
            'conv2d_64', 'batch_normalization_64', 'activation_64', 'conv2d_65', 'batch_normalization_65',
            'activation_65', 'conv2d_66', 'batch_normalization_66', 'activation_66', 'conv2d_67',
            'batch_normalization_67', 'activation_67', 'conv2d_68', 'batch_normalization_68', 'activation_68',
            'conv2d_61', 'batch_normalization_61', 'activation_61', 'conv2d_62', 'batch_normalization_62',
            'activation_62', 'conv2d_63', 'batch_normalization_63', 'activation_63', 'average_pooling2d_6', 'conv2d_69',
            'batch_normalization_69', 'activation_69', 'conv2d_60', 'batch_normalization_60', 'activation_60', 'mixed7'
        ]
    }
    shape = (299, 299, 3)

    def __init__(self, n: int, weights: Union[str, io] = None):
        super(InceptionV3Model, self).__init__(
            n=n, baseline=inception_v3.InceptionV3(include_top=False, weights=weights, input_shape=self.shape),
            preprocess_func=inception_v3.preprocess_input)


class DenseNetModel(GeneralModel):

    __name__ = 'DenseNet'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': [
            'conv5_block16_0_bn', 'conv5_block16_0_relu', 'conv5_block16_1_conv', 'conv5_block16_1_bn',
            'conv5_block16_1_relu', 'conv5_block16_2_conv', 'conv5_block16_concat', 'bn', 'relu'
        ],
        '2FT': [
            'conv5_block15_0_bn', 'conv5_block15_0_relu', 'conv5_block15_1_conv', 'conv5_block15_1_bn',
            'conv5_block15_1_relu', 'conv5_block15_2_conv', 'conv5_block15_concat'
        ],
        '3FT': [
            'conv5_block14_0_bn', 'conv5_block14_0_relu', 'conv5_block14_1_conv', 'conv5_block14_1_bn',
            'conv5_block14_1_relu', 'conv5_block14_2_conv', 'conv5_block14_concat'
        ],
        '4FT': [
            'conv5_block13_0_bn', 'conv5_block13_0_relu', 'conv5_block13_1_conv', 'conv5_block13_1_bn',
            'conv5_block13_1_relu', 'conv5_block13_2_conv', 'conv5_block13_concat'
        ]
    }
    shape = (224, 224, 3)

    def __init__(self, n: int, weights: Union[str, io] = None):
        super(DenseNetModel, self).__init__(
            n=n, baseline=densenet.DenseNet121(include_top=False, weights=weights, input_shape=self.shape),
            preprocess_func=densenet.preprocess_input
        )


class UnetVGG16Model(VGG16Model, GeneralModel):

    __name__ = 'UnetVGG16'
    loss = BinaryCrossentropy()
    metrics = [iou_score]

    def __init__(self, weights: Union[str, io] = None):
        super(UnetVGG16Model, self).__init__(
            n=0, baseline=Unet('vgg16', encoder_weights=weights, encoder_freeze=True),
            preprocess_func=vgg16.preprocess_input
        )

    def create_model(self):
        self.model = self.baseline


class UnetResnet50Model(Resnet50Model, GeneralModel):

    __name__ = 'UnetResnet50'
    loss = BinaryCrossentropy()

    def __init__(self, weights: Union[str, io] = None):
        super(UnetResnet50Model, self).__init__(
            n=0, baseline=Unet('resnet50', encoder_weights=weights, encoder_freeze=True),
            preprocess_func=resnet50.preprocess_input
        )

    def create_model(self):
        self.model = self.baseline


class UnetDenseNetModel(DenseNetModel, GeneralModel):

    __name__ = 'UnetDenseNet'
    loss = BinaryCrossentropy()

    def __init__(self, weights: Union[str, io] = None):
        super(UnetDenseNetModel, self).__init__(
            n=0, baseline=Unet('densenet121', encoder_weights=weights, encoder_freeze=True),
            preprocess_func=densenet.preprocess_input
        )

    def create_model(self):
        self.model = self.baseline


class UnetInceptionV3Model(InceptionV3Model, GeneralModel):

    __name__ = 'UnetInceptionV3'
    loss = BinaryCrossentropy()

    def __init__(self, weights: Union[str, io] = None):
        super(UnetInceptionV3Model, self).__init__(
            n=0, baseline=Unet('inceptionv3', encoder_weights=weights, encoder_freeze=True),
            preprocess_func=inception_v3.preprocess_input
        )

    def create_model(self):
        self.model = self.baseline