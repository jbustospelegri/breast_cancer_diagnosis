import numpy as np

from typing import Callable, io, Union, Tuple
from time import process_time

from tensorflow.keras import Model, Sequential, optimizers, callbacks
from tensorflow.keras.backend import count_params, eval
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.applications import resnet50, densenet, vgg16, inception_v3
from tensorflow.python.keras.preprocessing.image import DataFrameIterator
from tensorflow.keras.layers import (
    Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Input, BatchNormalization, Flatten
)

from segmentation_models import get_preprocessing

from utils.functions import get_path, get_number_of_neurons
from utils.config import CLASSIFICATION_LOSS


class GeneralModel:

    __name__ = 'GeneralModel'
    model: Model = None
    callbakcs = {}
    loss = CLASSIFICATION_LOSS
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
    BS_DICT = {
        '0FT': 32,
        '1FT': 32,
        '2FT': 32,
        '3FT': 32,
        '4FT': 32
    }

    def __init__(self, n: int = 1, baseline: Model = None, preprocess_func: Callable = None):
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

        #  Test con extract features y sigmoide
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)

        # Test añadiendo capas fully connected
        # x = Flatten()(x)
        # x = Dense(512, activation='relu')(x)
        # x = Dense(128, activation='relu')(x)
        # x = Dense(64, activation='relu', kernel_constraint=max_norm(3))(x)
        # x = Dropout(0.25)(x)
        # x = Dense(32, activation='relu', kernel_constraint=max_norm(3), activity_regularizer=L2(l2=0.01))(x)

        output = Dense(self.n, activation='softmax')(x)

        self.model = Model(inputs=input, outputs=output)

    def set_trainable_layers(self, unfrozen_layers: str):
        """
        Función utilizada para setear layers del modelo como entrenables. Este metodo será sobreescrito por las clases
        heredadas
        """
        self.baseline.trainable = True

        if unfrozen_layers == 'ALL':
            pass

        elif unfrozen_layers in self.LAYERS_DICT.keys():
            list_keys = sorted(self.LAYERS_DICT.keys(), key=lambda x: int(x[0]))
            train_layers = \
                [l for layers in list_keys[:list_keys.index(unfrozen_layers) + 1] for l in self.LAYERS_DICT[layers]]

            for layer in self.baseline.layers:
                if layer.name not in train_layers:
                    layer.trainable = False

        else:
            raise ValueError(f'Unfrozen layers parametrization for {unfrozen_layers}')

    def start_train(self, train_data: DataFrameIterator, val_data: DataFrameIterator, epochs: int,
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
            callbacks=list(self.callbakcs.values())
        )
        return process_time() - start, len(self.history.history['loss'])

    def register_metric(self, *args: Union[Callable, str]):
        for arg in args:
            self.metrics.append(arg)

    def register_callback(self, **kargs: callbacks):
        self.callbakcs = {**self.callbakcs, **kargs}

    def train_from_scratch(self, train_data, val_data, epochs: int, opt: optimizers = None):
        """
            Función utilizada para entrenar completamente el modelo.
        """
        t, e = self.start_train(train_data, val_data, epochs, opt, unfrozen_layers='ALL')
        return t, e

    def extract_features(self, train_data, val_data, epochs: int, opt: optimizers = None):
        """
        Función utilizada para aplicar un proceso de extract features de modo que se conjela la arquitectura definida en
        self.baseline y se entrenan las últimas capas de la arquitectura
        """
        t, e = self.start_train(train_data, val_data, epochs, opt, unfrozen_layers='0FT')
        return t, e

    def fine_tunning(self, train_data, val_data, epochs: int, opt: optimizers = None, unfrozen_layers: str = '1FT'):
        """
        Función utilizada para aplicar un proceso de transfer learning de modo que se conjelan n - k capas. Las k capas
        entrenables en la arquitectura definida por self.baseline se determinarán a partir del método
        set_trainable_layers
        """
        t, e = self.start_train(train_data, val_data, epochs, opt, unfrozen_layers=unfrozen_layers)
        return t, e

    def save_weights(self, dirname: str, model_name: str):
        """
        Función utilizada para almacenar el modelo entrenado
        :param dirname: directorio en el cual se almacenara el modelocv
        :param model_name: nombre del archivo para almacenar el modelo
        """
        self.model.save_weights(get_path(dirname, model_name))

    def load_weigths(self, weights: io):
        self.model.load_weights(weights)

    def predict(self, *args, **kwargs):
        """
        Función utilizada para generar las predicciones de un conjunto de datos dado
        """
        return self.model.predict(*args, **kwargs)

    def get_trainable_layers(self) -> int:
        return len([l for l in self.baseline.layers if l.trainable])

    def get_trainable_params(self) -> int:
        return int(np.sum([
            count_params(p) for lay in self.baseline.layers for p in lay.trainable_weights if lay.trainable
        ]))

    def get_learning_rate(self) -> float:
        return eval(self.model.optimizer.lr)


class VGG16Model(GeneralModel):

    __name__ = 'VGG16'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': ['block5_conv1', 'block5_conv2', 'block5_conv3'],
        '2FT': ['block4_conv1', 'block4_conv2', 'block4_conv3'],
        '3FT': ['block3_conv1', 'block3_conv2', 'block3_conv3'],
        '4FT': ['block2_conv1', 'block2_conv2']
    }
    BS_DICT = {
        '0FT': 96,
        '1FT': 88,
        '2FT': 80,
        '3FT': 72,
        '4FT': 64,
        'ALL': 40
    }

    def __init__(self, n: int, weights: Union[str, io] = None):
        super(VGG16Model, self).__init__(
            n=n, baseline=vgg16.VGG16(include_top=False, weights=weights, input_shape=self.shape),
            preprocess_func=vgg16.preprocess_input
        )

    def get_preprocessing_func(self):
        return get_preprocessing('vgg16')


class Resnet50Model(GeneralModel):

    __name__ = 'ResNet50'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': [
            'conv5_block3_1_conv', 'conv5_block3_2_conv', 'conv5_block3_3_conv', 'conv5_block2_1_conv',
            'conv5_block2_2_conv', 'conv5_block2_3_conv', 'conv5_block1_1_conv', 'conv5_block1_2_conv',
            'conv5_block1_0_conv', 'conv5_block1_3_conv', 'conv5_block1_1_bn', 'conv5_block1_2_bn',
            'conv5_block1_0_bn', 'conv5_block1_3_bn', 'conv5_block2_1_bn', 'conv5_block2_2_bn', 'conv5_block2_3_bn',
            'conv5_block3_1_bn', 'conv5_block3_2_bn', 'conv5_block3_3_bn'
        ],
        '2FT': [
            'conv4_block1_1_conv', 'conv4_block1_2_conv', 'conv4_block1_0_conv', 'conv4_block1_3_conv',
            'conv4_block2_1_conv', 'conv4_block2_2_conv', 'conv4_block2_3_conv', 'conv4_block3_1_conv',
            'conv4_block3_2_conv', 'conv4_block3_3_conv', 'conv4_block4_1_conv', 'conv4_block4_2_conv',
            'conv4_block4_3_conv', 'conv4_block5_1_conv', 'conv4_block5_2_conv', 'conv4_block5_3_conv',
            'conv4_block6_1_conv', 'conv4_block6_2_conv', 'conv4_block6_3_conv', 'conv4_block1_1_bn',
            'conv4_block1_2_bn', 'conv4_block1_0_bn', 'conv4_block1_3_bn', 'conv4_block2_1_bn', 'conv4_block2_2_bn',
            'conv4_block2_3_bn', 'conv4_block3_1_bn', 'conv4_block3_2_bn', 'conv4_block3_3_bn', 'conv4_block4_1_bn',
            'conv4_block4_2_bn', 'conv4_block4_3_bn', 'conv4_block5_1_bn', 'conv4_block5_2_bn', 'conv4_block5_3_bn',
            'conv4_block6_1_bn', 'conv4_block6_2_bn', 'conv4_block6_3_bn'
        ],
        '3FT': [
            'conv3_block1_1_conv', 'conv3_block1_2_conv', 'conv3_block1_0_conv', 'conv3_block1_3_conv',
            'conv3_block2_1_conv', 'conv3_block2_2_conv', 'conv3_block2_3_conv', 'conv3_block3_1_conv',
            'conv3_block3_2_conv', 'conv3_block3_3_conv', 'conv3_block4_1_conv', 'conv3_block4_2_conv',
            'conv3_block4_3_conv', 'conv3_block1_1_bn', 'conv3_block1_2_bn', 'conv3_block1_0_bn', 'conv3_block1_3_bn',
            'conv3_block2_1_bn', 'conv3_block2_2_bn', 'conv3_block2_3_bn', 'conv3_block3_1_bn', 'conv3_block3_2_bn',
            'conv3_block3_3_bn', 'conv3_block4_1_bn', 'conv3_block4_2_bn', 'conv3_block4_3_bn'
        ],
        '4FT': [
            'conv2_block1_1_conv', 'conv2_block1_2_conv', 'conv2_block1_0_conv', 'conv2_block1_3_conv',
            'conv2_block2_1_conv', 'conv2_block2_2_conv', 'conv2_block2_3_conv', 'conv2_block3_1_conv',
            'conv2_block3_2_conv', 'conv2_block3_3_conv', 'conv2_block1_1_bn', 'conv2_block1_2_bn', 'conv2_block1_0_bn',
            'conv2_block1_3_bn', 'conv2_block2_1_bn', 'conv2_block2_2_bn', 'conv2_block2_3_bn', 'conv2_block3_1_bn',
            'conv2_block3_2_bn', 'conv2_block3_3_bn'
        ]
    }
    BS_DICT = {
        '0FT': 128,
        '1FT': 128,
        '2FT': 112,
        '3FT': 56,
        '4FT': 32,
        'ALL': 28
    }
    shape = (224, 224, 3)

    def __init__(self, n: int, weights: Union[str, io] = None):
        super(Resnet50Model, self).__init__(
            n=n, baseline=resnet50.ResNet50(include_top=False, weights=weights, input_shape=self.shape),
            preprocess_func=resnet50.preprocess_input
        )

    def get_preprocessing_func(self):
        return get_preprocessing('resnet50')


class InceptionV3Model(GeneralModel):

    __name__ = 'InceptionV3'
    LAYERS_DICT = {
        '0FT': [],
        '1FT': [
            'conv2d_89', 'conv2d_90', 'conv2d_91', 'conv2d_92', 'conv2d_86', 'conv2d_87', 'conv2d_88', 'conv2d_93',
            'conv2d_85', 'batch_normalization_89', 'batch_normalization_90', 'batch_normalization_91',
            'batch_normalization_92', 'batch_normalization_86', 'batch_normalization_87', 'batch_normalization_88',
            'batch_normalization_93', 'batch_normalization_85'
        ],
        '2FT': [
            'conv2d_80', 'conv2d_81', 'conv2d_82', 'conv2d_83', 'conv2d_77', 'conv2d_78', 'conv2d_79', 'conv2d_84',
            'conv2d_76', 'batch_normalization_80', 'batch_normalization_81', 'batch_normalization_82',
            'batch_normalization_83', 'batch_normalization_77', 'batch_normalization_78', 'batch_normalization_79',
            'batch_normalization_84', 'batch_normalization_76',
        ],
        '3FT': [
            'conv2d_72', 'conv2d_73', 'conv2d_74', 'conv2d_75', 'conv2d_70', 'conv2d_71', 'batch_normalization_72',
            'batch_normalization_73', 'batch_normalization_74', 'batch_normalization_75', 'batch_normalization_70',
            'batch_normalization_71'
        ],
        '4FT': [
            'conv2d_64', 'conv2d_65', 'conv2d_66', 'conv2d_67', 'conv2d_68', 'conv2d_61', 'conv2d_62', 'conv2d_63',
            'conv2d_69', 'conv2d_60', 'batch_normalization_64', 'batch_normalization_65', 'batch_normalization_66',
            'batch_normalization_67', 'batch_normalization_68', 'batch_normalization_61', 'batch_normalization_62',
            'batch_normalization_63', 'batch_normalization_69', 'batch_normalization_60',
        ]
    }
    BS_DICT = {
        '0FT': 128,
        '1FT': 128,
        '2FT': 128,
        '3FT': 128,
        '4FT': 128,
        'ALL': 24
    }
    shape = (299, 299, 3)

    def __init__(self, n: int, weights: Union[str, io] = None):
        super(InceptionV3Model, self).__init__(
            n=n, baseline=inception_v3.InceptionV3(include_top=False, weights=weights, input_shape=self.shape),
            preprocess_func=inception_v3.preprocess_input)

    def get_preprocessing_func(self):
        return get_preprocessing('inceptionv3')


class DenseNetModel(GeneralModel):

    __name__ = 'DenseNet'
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
            'conv5_block15_1_conv', 'conv5_block15_2_conv', 'conv5_block16_1_conv', 'conv5_block16_2_conv',
            'conv5_block1_0_bn', 'conv5_block1_1_bn', 'conv5_block2_0_bn', 'conv5_block2_1_bn', 'conv5_block3_0_bn',
            'conv5_block3_1_bn', 'conv5_block4_0_bn', 'conv5_block4_1_bn', 'conv5_block5_0_bn', 'conv5_block5_1_bn',
            'conv5_block6_0_bn', 'conv5_block6_1_bn', 'conv5_block7_0_bn', 'conv5_block7_1_bn', 'conv5_block8_0_bn',
            'conv5_block8_1_bn', 'conv5_block9_0_bn', 'conv5_block9_1_bn', 'conv5_block10_0_bn', 'conv5_block10_1_bn',
            'conv5_block11_0_bn', 'conv5_block11_1_bn', 'conv5_block12_0_bn', 'conv5_block12_1_bn',
            'conv5_block13_0_bn', 'conv5_block13_1_bn', 'conv5_block14_0_bn', 'conv5_block14_1_bn',
            'conv5_block15_0_bn', 'conv5_block15_1_bn', 'conv5_block16_0_bn', 'conv5_block16_1_bn', 'bn'
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
            'conv4_block23_1_conv', 'conv4_block23_2_conv', 'conv4_block24_1_conv', 'conv4_block24_2_conv',
            'conv4_block1_0_bn', 'conv4_block1_1_bn', 'conv4_block2_0_bn', 'conv4_block2_1_bn', 'conv4_block3_0_bn',
            'conv4_block3_1_bn', 'conv4_block4_0_bn', 'conv4_block4_1_bn', 'conv4_block5_0_bn', 'conv4_block5_1_bn',
            'conv4_block6_0_bn', 'conv4_block6_1_bn', 'conv4_block7_0_bn', 'conv4_block7_1_bn', 'conv4_block8_0_bn',
            'conv4_block8_1_bn', 'conv4_block9_0_bn', 'conv4_block9_1_bn', 'conv4_block10_0_bn', 'conv4_block10_1_bn',
            'conv4_block11_0_bn', 'conv4_block11_1_bn', 'conv4_block12_0_bn', 'conv4_block12_1_bn',
            'conv4_block13_0_bn', 'conv4_block13_1_bn', 'conv4_block14_0_bn', 'conv4_block14_1_bn',
            'conv4_block15_0_bn', 'conv4_block15_1_bn', 'conv4_block16_0_bn', 'conv4_block16_1_bn',
            'conv4_block17_0_bn', 'conv4_block17_1_bn', 'conv4_block18_0_bn', 'conv4_block18_1_bn',
            'conv4_block19_0_bn', 'conv4_block19_1_bn', 'conv4_block20_0_bn', 'conv4_block20_1_bn',
            'conv4_block21_0_bn', 'conv4_block21_1_bn', 'conv4_block22_0_bn', 'conv4_block22_1_bn',
            'conv4_block23_0_bn', 'conv4_block23_1_bn', 'conv4_block24_0_bn', 'conv4_block24_1_bn',
            'pool2_bn', 'pool2_conv'
        ],
        '3FT': [
            'conv3_block1_1_conv', 'conv3_block1_2_conv', 'conv3_block2_1_conv', 'conv3_block2_2_conv',
            'conv3_block3_1_conv', 'conv3_block3_2_conv', 'conv3_block4_1_conv', 'conv3_block4_2_conv',
            'conv3_block5_1_conv', 'conv3_block5_2_conv', 'conv3_block6_1_conv', 'conv3_block6_2_conv',
            'conv3_block7_1_conv', 'conv3_block7_2_conv', 'conv3_block8_1_conv', 'conv3_block8_2_conv',
            'conv3_block9_1_conv', 'conv3_block9_2_conv', 'conv3_block10_1_conv', 'conv3_block10_2_conv',
            'conv3_block11_1_conv', 'conv3_block11_2_conv', 'conv3_block12_1_conv', 'conv3_block12_2_conv',
            'conv3_block1_0_bn', 'conv3_block1_1_bn', 'conv3_block2_0_bn', 'conv3_block2_1_bn', 'conv3_block3_0_bn',
            'conv3_block3_1_bn', 'conv3_block4_0_bn', 'conv3_block4_1_bn', 'conv3_block5_0_bn', 'conv3_block5_1_bn',
            'conv3_block6_0_bn', 'conv3_block6_1_bn', 'conv3_block7_0_bn', 'conv3_block7_1_bn', 'conv3_block8_0_bn',
            'conv3_block8_1_bn', 'conv3_block9_0_bn', 'conv3_block9_1_bn', 'conv3_block10_0_bn', 'conv3_block10_1_bn',
            'conv3_block11_0_bn', 'conv3_block11_1_bn', 'conv3_block12_0_bn', 'conv3_block12_1_bn',
            'pool3_bn', 'pool3_conv'
        ],
        '4FT': [
            'conv2_block1_1_conv', 'conv2_block1_2_conv', 'conv2_block2_1_conv', 'conv2_block2_2_conv',
            'conv2_block3_1_conv', 'conv2_block3_2_conv', 'conv2_block4_1_conv', 'conv2_block4_2_conv',
            'conv2_block5_1_conv', 'conv2_block5_2_conv', 'conv2_block6_1_conv', 'conv2_block6_2_conv',
            'conv2_block1_0_bn', 'conv2_block1_1_bn', 'conv2_block2_0_bn', 'conv2_block2_1_bn', 'conv2_block3_0_bn',
            'conv2_block3_1_bn', 'conv2_block4_0_bn', 'conv2_block4_1_bn', 'conv2_block5_0_bn', 'conv2_block5_1_bn',
            'conv2_block6_0_bn', 'conv2_block6_1_bn', 'pool4_bn', 'pool4_conv'
        ]
    }
    BS_DICT = {
        '0FT': 128,
        '1FT': 128,
        '2FT': 26,
        '3FT': 24,
        '4FT': 22,
        'ALL': 20
    }

    def __init__(self, n: int, weights: Union[str, io] = None):
        super(DenseNetModel, self).__init__(
            n=n, baseline=densenet.DenseNet121(include_top=False, weights=weights, input_shape=self.shape),
            preprocess_func=densenet.preprocess_input
        )

    def get_preprocessing_func(self):
        return get_preprocessing('densenet121')
