import keras
import os

from typing import Callable, io, Union
from keras import Model, Sequential
from time import time

from keras.callbacks import History
from keras.optimizers import Adam
from keras.applications import resnet50, densenet, vgg16, inception_v3
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import DataFrameIterator


class GeneralModel:

    __name__ = 'GeneralModel'
    model: Model = None
    callbakcs = {}
    metrics = ['accuracy']
    history: History = None
    layers_dict = {
        '1FT': ['block4_dropout', 'block4_maxpool', 'block4_bn3', 'block4_conv2', 'block4_bn1', 'block4_conv1'],
        '2FT': ['block3_dropout', 'block3_maxpool', 'block3_bn3', 'block3_conv2', 'block3_bn1', 'block3_conv1'],
        '3FT': ['block2_dropout', 'block2_maxpool', 'block2_bn2', 'block2_conv2', 'block2_bn1', 'block2_conv1'],
        '4FT': ['block1_dropout', 'block1_maxpool', 'block1_bn1', 'block1_conv1']
    }

    def __init__(self, n_clases: int, baseline: keras.Model = None, input_shape: tuple = (250, 250),
                 preprocess_func: Callable = None, weights: str = None):
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
        out = Dropout(0.5)(out)
        predictions = Dense(self.n_clases, activation='softmax', kernel_regularizer=keras.regularizers.L2())(out)

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
                self.model.get_layer(layer.name).trainable = layer.name in self.baseline.layers
        elif unfrozen_layers in self.layers_dict.keys():
            list_keys = sorted(self.layers_dict.keys(), key=lambda x: int(x[0]))
            for layer_names in [self.layers_dict[d] for d in list_keys[:list_keys.index(unfrozen_layers) + 1]]:
                for layer in layer_names:
                    self.model.get_layer(name=layer).trainable = True

    def __start_train(self, train_data: DataFrameIterator, val_data: DataFrameIterator, epochs: int, batch_size: int,
                      opt: keras.optimizers = None, unfrozen_layers: str = 'ALL'):
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

        self.history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            verbose=2,
            callbacks=list(self.callbakcs.values()),
            steps_per_epoch=train_data.samples // batch_size,
            validation_steps=val_data.samples // batch_size,
        )

    def register_metric(self, *args: Union[Callable, str]):
        for arg in args:
            self.metrics.append(arg)

    def register_callback(self, **kargs: keras.callbacks):
        self.callbakcs = {**self.callbakcs, **kargs}

    def train_from_scratch(self, train_data, val_data, epochs: int, batch_size: int, opt: keras.optimizers = None):
        """
            Función utilizada para entrenar completamente el modelo.
        """
        self.__start_train(train_data, val_data, epochs, batch_size, opt, unfrozen_layers='ALL')

    def extract_features(self, train_data, val_data, epochs: int, batch_size: int, opt: keras.optimizers = None):
        """
        Función utilizada para aplicar un proceso de extract features de modo que se conjela la arquitectura definida en
        self.baseline y se entrenan las últimas capas de la arquitectura
        """
        self.__start_train(train_data, val_data, epochs, batch_size, opt, unfrozen_layers='0FT')

    def fine_tunning(self, train_data, val_data, epochs: int, batch_size: int, opt: keras.optimizers = None,
                     unfrozen_layers: str = '1FT'):
        """
        Función utilizada para aplicar un proceso de transfer learning de modo que se conjelan n - k capas. Las k capas
        entrenables en la arquitectura definida por self.baseline se determinarán a partir del método
        set_trainable_layers
        """
        self.__start_train(train_data, val_data, epochs, batch_size, opt, unfrozen_layers=unfrozen_layers)

    def save_model(self, dirname: str, model_name: str):
        """
        Función utilizada para almacenar el modelo entrenado
        :param dirname: directorio en el cual se almacenara el modelo
        :param model_name: nombre del archivo para almacenar el modelo
        """
        self.model.save(os.path.join(dirname, model_name))

    def predict(self, *args, **kwargs):
        """
        Función utilizada para generar las predicciones de un conjunto de datos dado
        """
        return self.model.predict(*args, **kwargs)


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

    def __init__(self, n_clases: int, weights: Union[str, io] = None):
        super().__init__(
            n_clases=n_clases, preprocess_func=resnet50.preprocess_input, input_shape=(224, 224),
            baseline=resnet50.ResNet50(include_top=False, weights=weights, input_shape=(224, 224, 3))
        )

    def set_trainable_layers(self):
        for layer in self.baseline.layers:
            layer.trainable = ('conv5_block3_' in layer.name) or ('conv5_block2_' in layer.name)


class InceptionV3Model(GeneralModel):

    __name__ = 'InceptionV3'

    def __init__(self, n_clases: int, weights: Union[str, io] = None):
        super().__init__(
            n_clases=n_clases, preprocess_func=inception_v3.preprocess_input, input_shape=(299, 299),
            baseline=inception_v3.InceptionV3(include_top=False, weights=weights, input_shape=(299, 299, 3)))

    def set_trainable_layers(self):
        self.baseline.get_layer('conv2d_87').trainable = True
        self.baseline.get_layer('conv2d_88').trainable = True
        self.baseline.get_layer('conv2d_91').trainable = True
        self.baseline.get_layer('conv2d_92').trainable = True
        self.baseline.get_layer('average_pooling2d_8').trainable = True
        self.baseline.get_layer('conv2d_85').trainable = True
        self.baseline.get_layer('batch_normalization_87').trainable = True
        self.baseline.get_layer('batch_normalization_88').trainable = True
        self.baseline.get_layer('batch_normalization_91').trainable = True
        self.baseline.get_layer('batch_normalization_92').trainable = True
        self.baseline.get_layer('conv2d_93').trainable = True
        self.baseline.get_layer('batch_normalization_85').trainable = True
        self.baseline.get_layer('activation_87').trainable = True
        self.baseline.get_layer('activation_88').trainable = True
        self.baseline.get_layer('activation_91').trainable = True
        self.baseline.get_layer('activation_92').trainable = True
        self.baseline.get_layer('batch_normalization_93').trainable = True
        self.baseline.get_layer('activation_85').trainable = True
        self.baseline.get_layer('mixed9_1').trainable = True
        self.baseline.get_layer('concatenate_1').trainable = True
        self.baseline.get_layer('activation_93').trainable = True
        self.baseline.get_layer('mixed10').trainable = True


class DenseNetModel(GeneralModel):

    __name__ = 'DenseNet'

    def __init__(self, n_clases: int, weights: Union[str, io] = None):
        super().__init__(
            n_clases=n_clases, preprocess_func=densenet.preprocess_input, input_shape=(224, 224),
            baseline=densenet.DenseNet121(include_top=False, weights=weights, input_shape=(224, 224, 3)))

    def set_trainable_layers(self):
        for layer in self.baseline.layers:
            # Posibilidad de poner tambien conv5_ directamente
            layer.trainable = ('conv5_block16' in layer.name) or ('conv5_block15' in layer.name) or \
                              ('conv5_block14' in layer.name)
        self.baseline.get_layer('bn').trainable = True
        self.baseline.get_layer('relu').trainable = True


