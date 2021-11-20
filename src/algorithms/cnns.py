from typing import Callable, io, Union
from keras import Model, Sequential, optimizers, callbacks
from time import process_time

from keras.callbacks import History
from keras.optimizers import Adam
from keras.regularizers import L2
from keras.applications import resnet50, densenet, vgg16, inception_v3
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import DataFrameIterator

from utils.functions import get_path, get_number_of_neurons


class GeneralModel:

    __name__ = 'GeneralModel'
    model: Model = None
    callbakcs = {}
    metrics = ['accuracy']
    history: History = None
    LAYERS_DICT = {
        '1FT': ['block4_dropout', 'block4_maxpool', 'block4_bn3', 'block4_conv2', 'block4_bn1', 'block4_conv1'],
        '2FT': ['block3_dropout', 'block3_maxpool', 'block3_bn3', 'block3_conv2', 'block3_bn1', 'block3_conv1'],
        '3FT': ['block2_dropout', 'block2_maxpool', 'block2_bn2', 'block2_conv2', 'block2_bn1', 'block2_conv1'],
        '4FT': ['block1_dropout', 'block1_maxpool', 'block1_bn1', 'block1_conv1']
    }

    def __init__(self, n_clases: int, baseline: Model = None, input_shape: tuple = (250, 250), weights: str = None,
                 preprocess_func: Callable = None):
        self.baseline = baseline
        self.n_clases = n_clases
        self.input_shape = input_shape
        self.preprocess_func = preprocess_func
        self.weights = weights
        self.__create_model()

    def __create_baseline(self):
        """
        Función que permite crear una estructura básica compuesta por un conjunto de capas convolucionales. Este metodo
        será sobreescrito por las clases heredadas.
        """
        self.baseline = Sequential()

        self.baseline.add(Conv2D(32, (3, 3), padding="same", input_shape=(*self.input_shape, 3), activation='relu',
                                 name='block1_conv1'))
        self.baseline.add(BatchNormalization(axis=1, name='block1_bn1'))
        self.baseline.add(MaxPooling2D(pool_size=(3, 3), name='block1_maxpool'))
        self.baseline.add(Dropout(0.25, name='block1_dropout'))

        self.baseline.add(Conv2D(64, (3, 3), padding="same", activation='relu', name='block2_conv1'))
        self.baseline.add(BatchNormalization(axis=1, name='block2_bn1'))
        self.baseline.add(Conv2D(64, (3, 3), padding="same", activation='relu', name='block2_conv2'))
        self.baseline.add(BatchNormalization(axis=1, name='block2_bn2'))
        self.baseline.add(MaxPooling2D(pool_size=(2, 2), name='block2_maxpool'))
        self.baseline.add(Dropout(0.25, name='block2_dropout'))

        self.baseline.add(Conv2D(128, (3, 3), padding="same", activation='relu', name='block3_conv1'))
        self.baseline.add(BatchNormalization(axis=1, name='block3_bn1'))
        self.baseline.add(Conv2D(128, (3, 3), padding="same", activation='relu', name='block3_conv2'))
        self.baseline.add(BatchNormalization(axis=1, name='block3_bn3'))
        self.baseline.add(MaxPooling2D(pool_size=(2, 2), name='block3_maxpool'))
        self.baseline.add(Dropout(0.25, name='block3_dropout'))

        self.baseline.add(Conv2D(256, (3, 3), padding="same", activation='relu', name='block4_conv1'))
        self.baseline.add(BatchNormalization(axis=1, name='block4_bn1'))
        self.baseline.add(Conv2D(256, (3, 3), padding="same", activation='relu', name='block4_conv2'))
        self.baseline.add(BatchNormalization(axis=1, name='block4_bn3'))
        self.baseline.add(MaxPooling2D(pool_size=(2, 2), name='block4_maxpool'))
        self.baseline.add(Dropout(0.25, name='block4_dropout'))

    def __create_model(self):
        """
        Función utilizada para crear la estructura de un modelo. Esta, estará formada por la estructura básica devuelta
        por self.baseline juntamente con dos capas FC de 512 neuronas con función de activación relu. La capa de salida
        estará compuesta por una capa de salida FC con tantas neuronas como clases existan en el dataset y con función
        de activación softmax
        """

        if self.baseline is None:
            self.__create_baseline()

        out = self.baseline.output
        out = GlobalAveragePooling2D()(out)

        neurons = get_number_of_neurons(out.get_shape().as_list())
        while neurons > 15:
            out = Dropout(0.5)(out)
            out = Dense(neurons, activation='relu', kernel_regularizer=L2())(out)
            neurons = get_number_of_neurons(out.get_shape().as_list())

        out = Dropout(0.5)(out)
        predictions = Dense(self.n_clases, activation='softmax', kernel_regularizer=L2())(out)

        self.model = Model(inputs=self.baseline.input, outputs=predictions)

    def __set_trainable_layers(self, unfrozen_layers: str):
        """
        Función utilizada para setear layers del modelo como entrenables. Este metodo será sobreescrito por las clases
        heredadas
        """
        if unfrozen_layers == 'ALL':
            self.model.trainable = True
        elif unfrozen_layers == '0FT':
            for layer in self.model.layers:
                layer.trainable = layer.name not in [l.name for l in self.baseline.layers]
        elif unfrozen_layers in self.LAYERS_DICT.keys():
            self.__set_trainable_layers(unfrozen_layers='0FT')
            list_keys = sorted(self.LAYERS_DICT.keys(), key=lambda x: int(x[0]))
            train_layers = [l for layers in list_keys[:list_keys.index(unfrozen_layers) + 1] for l
                            in self.LAYERS_DICT[layers]]
            for layer in train_layers:
                self.model.get_layer(layer).trainable = True

        else:
            raise ValueError(f'Unfrozen layers parametrization for {unfrozen_layers}')

    def __start_train(self, train_data: DataFrameIterator, val_data: DataFrameIterator, epochs: int, batch_size: int,
                      opt: optimizers = None, unfrozen_layers: str = 'ALL') -> float:
        """
        Función que compila el modelo, añade los callbacks definidos por el usuario y entrena el modelo
        :param train_data: dataframe iterator con los datos de train
        :param val_data: dataframe iterator con los datos de validación
        :param epochs: número de épocas con las que realizar el entrenamiento
        :param batch_size: tamaño del batch
        :param opt: algorítmo de optimización de gradiente descendente

        """

        # Se configura si los layers serán entrenables o no
        self.model.trainable = False
        self.__set_trainable_layers(unfrozen_layers=unfrozen_layers)

        # Se compila el modelo
        self.model.compile(
            optimizer=opt or Adam(lr=1e-3),
            loss='categorical_crossentropy' if self.n_clases > 2 else 'binary_crossentropy',
            metrics=self.metrics
        )

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
        return process_time() - start

    def register_metric(self, *args: Union[Callable, str]):
        for arg in args:
            self.metrics.append(arg)

    def register_callback(self, **kargs: callbacks):
        self.callbakcs = {**self.callbakcs, **kargs}

    def train_from_scratch(self, train_data, val_data, epochs: int, batch_size: int, opt: optimizers = None):
        """
            Función utilizada para entrenar completamente el modelo.
        """
        t = self.__start_train(train_data, val_data, epochs, batch_size, opt, unfrozen_layers='ALL')
        return t

    def extract_features(self, train_data, val_data, epochs: int, batch_size: int, opt: optimizers = None):
        """
        Función utilizada para aplicar un proceso de extract features de modo que se conjela la arquitectura definida en
        self.baseline y se entrenan las últimas capas de la arquitectura
        """
        t = self.__start_train(train_data, val_data, epochs, batch_size, opt, unfrozen_layers='0FT')
        return t

    def fine_tunning(self, train_data, val_data, epochs: int, batch_size: int, opt: optimizers = None,
                     unfrozen_layers: str = '1FT'):
        """
        Función utilizada para aplicar un proceso de transfer learning de modo que se conjelan n - k capas. Las k capas
        entrenables en la arquitectura definida por self.baseline se determinarán a partir del método
        set_trainable_layers
        """
        t = self.__start_train(train_data, val_data, epochs, batch_size, opt, unfrozen_layers=unfrozen_layers)
        return t

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
        return len([l for l in self.model.layers if l.trainable]) - 1


class VGG16Model(GeneralModel):

    __name__ = 'VGG16'
    LAYERS_DICT = {
        '1FT': ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool'],
        '2FT': ['block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool'],
        '3FT': ['block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool'],
        '4FT': ['block2_conv1', 'block2_conv2', 'block2_pool']
    }

    def __init__(self, n_clases: int, weights: Union[str, io] = None):
        super().__init__(
            n_clases=n_clases, preprocess_func=vgg16.preprocess_input, input_shape=(224, 224),
            baseline=vgg16.VGG16(include_top=False, weights=weights, input_shape=(224, 224, 3))
        )


class Resnet50Model(GeneralModel):

    __name__ = 'ResNet50'
    LAYERS_DICT = {
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

    def __init__(self, n_clases: int, weights: Union[str, io] = None):
        super().__init__(
            n_clases=n_clases, preprocess_func=resnet50.preprocess_input, input_shape=(224, 224),
            baseline=resnet50.ResNet50(include_top=False, weights=weights, input_shape=(224, 224, 3))
        )


class InceptionV3Model(GeneralModel):

    __name__ = 'InceptionV3'
    LAYERS_DICT = {
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

    def __init__(self, n_clases: int, weights: Union[str, io] = None):
        super().__init__(
            n_clases=n_clases, preprocess_func=inception_v3.preprocess_input, input_shape=(299, 299),
            baseline=inception_v3.InceptionV3(include_top=False, weights=weights, input_shape=(299, 299, 3)))


class DenseNetModel(GeneralModel):

    __name__ = 'DenseNet'
    LAYERS_DICT = {
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

    def __init__(self, n_clases: int, weights: Union[str, io] = None):
        super().__init__(
            n_clases=n_clases, preprocess_func=densenet.preprocess_input, input_shape=(224, 224),
            baseline=densenet.DenseNet121(include_top=False, weights=weights, input_shape=(224, 224, 3)))
