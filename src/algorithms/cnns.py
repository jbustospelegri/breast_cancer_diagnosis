import keras
import os
import numpy as np

from typing import Callable, io, Union
from keras import Model, Sequential
from time import time

from src.utils.functions import log_cmd_to_file

from keras.callbacks import EarlyStopping, CSVLogger, History
from keras.optimizers import Adam
from keras.applications import resnet50, densenet, vgg16, inception_v3
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D


class GeneralModel:

    __name__ = 'GeneralModel'
    model: Model = None
    optimizer: keras.optimizers = None
    callbakcs = []
    history: History = None

    def __init__(self, n_clases: int, log_dir: io, summary_dir: io,  test_name: str = '', baseline: keras.Model = None,
                 input_shape: tuple = (250, 250), preprocess_func: Callable = None, get_model_structure: bool = False):
        self.baseline = baseline
        self.n_clases = n_clases
        self.input_shape = input_shape
        self.test_name = test_name
        self.preprocess_func = preprocess_func
        self.log_dir = log_dir
        self.log_filename = os.path.join(log_dir, f'{"_".join([self.__name__, self.test_name])}.csv')
        self.summary_dir = summary_dir
        self.summary_filename = os.path.join(summary_dir, f'{"_".join([self.__name__, self.test_name])}.txt')
        self.create_model(print_model=get_model_structure)
        self.metrics = ['accuracy']

    def register_metric(self, *args: Union[Callable, str]):
        for arg in args:
            self.metrics.append(arg)

    @log_cmd_to_file
    def repr_summary(self, **kwargs):
        """
        Función utilizada para representar el summary de un modelo, el número de parámetros entrenables y un conjunto
        de parámetros definidos por el usuario a partir del kwarg txt_kwargs
        """

        if kwargs.get('txt_kwargs', None):
            print('-' * 50)
            print('\tInformación Adicional')
            print('\n\t\t- '.join([f'{k}: {v}' for k, v in kwargs.get('txt_kwargs', {}).items()]))
            print('-' * 50)

        if kwargs.get('get_summary', False):
            print('\n\tInformación del modelo:\n')
            self.model.summary()
            print('-' * 50)

        elif kwargs.get('get_number_params', False):
            trainable_count = int(np.sum([keras.backend.count_params(p) for p in self.model.trainable_weights]))
            non_trainable_count = int(np.sum([keras.backend.count_params(p) for p in self.model.non_trainable_weights]))
            print('\tInformación parámeros del modelo:')
            print('\t\t-Número total de parámetros: {:,}'.format(trainable_count + non_trainable_count))
            print('\t\t-Parámetros entrenables: {:,}'.format(trainable_count))
            print('\t\t-Parámetros no entrenables: {:,}'.format(non_trainable_count))

    def create_baseline(self):
        """
        Función que permite crear una estructura básica compuesta por un conjunto de capas convolucionales. Este metodo
        será sobreescrito por las clases heredadas.
        """
        self.baseline = Sequential()

        self.baseline.add(Conv2D(32, (3, 3), padding="same", input_shape=(*self.input_shape, 3), activation='relu'))
        self.baseline.add(BatchNormalization(axis=1))
        self.baseline.add(MaxPooling2D(pool_size=(3, 3)))
        self.baseline.add(Dropout(0.25))

        self.baseline.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
        self.baseline.add(BatchNormalization(axis=1))
        self.baseline.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
        self.baseline.add(BatchNormalization(axis=1))
        self.baseline.add(MaxPooling2D(pool_size=(2, 2)))
        self.baseline.add(Dropout(0.25))

        self.baseline.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
        self.baseline.add(BatchNormalization(axis=1))
        self.baseline.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
        self.baseline.add(BatchNormalization(axis=1))
        self.baseline.add(MaxPooling2D(pool_size=(2, 2)))
        self.baseline.add(Dropout(0.25))

    def create_model(self, print_model: bool = False):
        """
        Función utilizada para crear la estructura de un modelo. Esta, estará formada por la estructura básica devuelta
        por self.baseline juntamente con dos capas FC de 512 neuronas con función de activación relu. La capa de salida
        estará compuesta por una capa de salida FC con tantas neuronas como clases existan en el dataset y con función
        de activación softmax
        :param print_model: booleano que permite recuperar la estructura del modelo compilado
        """
        if self.baseline is None:
            self.create_baseline()

        out = self.baseline.output
        out = GlobalAveragePooling2D()(out)
        out = Dense(512, activation='relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(512, activation='relu')(out)
        out = Dropout(0.5)(out)
        predictions = Dense(self.n_clases, activation='softmax')(out)

        self.model = Model(inputs=self.baseline.input, outputs=predictions)

        self.repr_summary(get_summary=print_model, dirname=self.summary_dir, file_log=self.summary_filename)

    def compile_model(self, opt: keras.optimizers = None):
        """
        Función utilizada para compilar un modelo. La función de pérdidas será categorical_crossentropy y las métricas
        serán accuracy y f1_score.
        :param opt: optimizador de keras con el que se compilará el modelo. Por defecto será Adam con learning rate de
                    1e-3
        """

        if opt is None:
            self.optimizer = Adam(lr=1e-3)
        else:
            self.optimizer = opt

        if self.n_clases > 2:
            self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=self.metrics)
        else:
            self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=self.metrics)

    def set_model_callbacks(self, early_stopping: bool = True, csv_logger: bool = True):
        """
        Función utilizada para añadir callbacks a la fase de entrenamiento de un modelo
        :param early_stopping: booleano que permite añadir un callback de early stopping
        :param csv_logger:  boleano que permite añadir un callback de csvlogger
        """

        self.callbakcs = []

        if early_stopping:
            self.callbakcs.append(EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True))

        if csv_logger:
            self.callbakcs.append(CSVLogger(filename=self.log_filename, separator=';', append=True))

    def set_trainable_layers(self):
        """
        Función utilizada para setear layers del modelo como entrenables. Este metodo será sobreescrito por las clases
        heredadas
        """
        pass

    def train_pipe(self, train_data, val_data, epochs: int, batch_size: int, opt: keras.optimizers = None,
                   **model_callbacks):
        """
        Función que compila el modelo, añade los callbacks definidos por el usuario y entrena el modelo
        :param train_data: dataframe iterator con los datos de train
        :param val_data: dataframe iterator con los datos de validación
        :param epochs: número de épocas con las que realizar el entrenamiento
        :param batch_size: tamaño del batch
        :param opt: algorítmo de optimización de gradiente descendente
        :param model_callbacks: kwargs compuesto por booleanos que permiten al usuario añadir callbacks al proceso de
                                entrenamiento
        """

        self.compile_model(opt)

        self.set_model_callbacks(model_callbacks.get('early_stopping', True), model_callbacks.get('csv_logger', True))

        self.history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            verbose=2,
            callbacks=self.callbakcs,
            steps_per_epoch=train_data.samples // batch_size,
            validation_steps=val_data.samples // batch_size,
        )

    def train_from_scratch_pipe(self, train_data, val_data, epochs: int, batch_size: int, opt: keras.optimizers = None,
                                **model_callbacks):
        """
            Función utilizada para entrenar completamente el modelo.
        """
        self.model.trainable = True
        start_exec = time()
        self.train_pipe(train_data, val_data, epochs, batch_size, opt, **model_callbacks)
        # se almacenan un conjunto de métricas obtenidas de la fase de entrenamiento.
        self.repr_summary(file_log=self.summary_filename,
                          txt_kwargs={
                              '\n' + '-' * 50 + '\n\tExecution': 'Train from scratch\n' + '-' * 50,
                              'Batch size': batch_size,
                              'Optimizador': opt.get_config()['name'],
                              'Learning Rate': opt.get_config()['learning_rate'],
                              'Epochs': f'{len(self.history.history["loss"])} of {epochs}',
                              'Tiempo ejecucion': f'{(time() - start_exec) / 60:.2f} minutos'
                          })

    def extract_features_pipe(self, train_data, val_data, epochs: int, batch_size: int, opt: keras.optimizers = None,
                              **model_callbacks):
        """
        Función utilizada para aplicar un proceso de extract features de modo que se conjela la arquitectura definida en
        self.baseline y se entrenan las últimas capas de la arquitectura
        """

        self.baseline.trainable = False
        start_exec = time()
        self.train_pipe(train_data, val_data, epochs, batch_size, opt, **model_callbacks)
        # se almacenan un conjunto de métricas obtenidas de la fase de entrenamiento.
        self.repr_summary(file_log=self.summary_filename,
                          get_number_params=True,
                          txt_kwargs={
                              '\n' + '-' * 50 + '\n\tExecution': 'Extract - Features\n' + '-' * 50,
                              'Batch size': batch_size,
                              'Optimizador': opt.get_config()['name'],
                              'Learning Rate': opt.get_config()['learning_rate'],
                              'Epochs': f'{len(self.history.history["loss"])} of {epochs}',
                              'Tiempo ejecucion': f'{(time() - start_exec) / 60:.2f} minutos'
                           })

    def transfer_learning_pipe(self, train_data, val_data, epochs: int, batch_size: int, opt: keras.optimizers = None,
                               **model_callbacks):
        """
        Función utilizada para aplicar un proceso de transfer learning de modo que se conjelan n - k capas. Las k capas
        entrenables en la arquitectura definida por self.baseline se determinarán a partir del método
        set_trainable_layers
        """

        self.baseline.trainable = False
        self.set_trainable_layers()

        start_exec = time()
        self.train_pipe(train_data, val_data, epochs, batch_size, opt, **model_callbacks)
        # se almacenan un conjunto de métricas obtenidas de la fase de entrenamiento.
        self.repr_summary(file_log=self.summary_filename,
                          get_number_params=True,
                          txt_kwargs={
                              '\n' + '-' * 50 + '\n\tExecution': 'Transfer Learning\n' + '-' * 50,
                              'Batch size': batch_size,
                              'Optimizador': opt.get_config()['name'],
                              'Learning Rate': opt.get_config()['learning_rate'],
                              'Epochs': f'{len(self.history.history["loss"])} of {epochs}',
                              'Tiempo ejecucion': f'{(time() - start_exec) / 60:.2f} minutos'
                          })

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

    def __init__(self, n_clases: int, log_dir: io, summary_dir: io, test_name: str = '',
                 get_model_structure: bool = False):
        super().__init__(
            n_clases=n_clases, preprocess_func=vgg16.preprocess_input, input_shape=(224, 224), test_name=test_name,
            get_model_structure=get_model_structure, summary_dir=summary_dir, log_dir=log_dir,
            baseline=vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        )

    def set_trainable_layers(self):
        for layer in self.baseline.layers:
            layer.trainable = 'block5' in layer.name


class Resnet50Model(GeneralModel):

    __name__ = 'ResNet50'

    def __init__(self, n_clases: int, log_dir: io, summary_dir: io, test_name: str = '',
                 get_model_structure: bool = False):
        super().__init__(
            n_clases=n_clases, preprocess_func=resnet50.preprocess_input, log_dir=log_dir, test_name=test_name,
            summary_dir=summary_dir, get_model_structure=get_model_structure, input_shape=(224, 224),
            baseline=resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        )

    def set_trainable_layers(self):
        for layer in self.baseline.layers:
            layer.trainable = ('conv5_block3_' in layer.name) or ('conv5_block2_' in layer.name)


class InceptionV3Model(GeneralModel):

    __name__ = 'InceptionV3'

    def __init__(self, n_clases: int, log_dir: io, summary_dir: io, test_name: str = '',
                 get_model_structure: bool = False):
        super().__init__(
            n_clases=n_clases, preprocess_func=inception_v3.preprocess_input, input_shape=(299, 299),
            test_name=test_name, get_model_structure=get_model_structure, summary_dir=summary_dir, log_dir=log_dir,
            baseline=inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3)))

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

    def __init__(self, n_clases: int, log_dir: io, summary_dir: io, test_name: str = '',
                 get_model_structure: bool = False):
        super().__init__(
            n_clases=n_clases, preprocess_func=densenet.preprocess_input, input_shape=(224, 224),
            test_name=test_name, get_model_structure=get_model_structure, summary_dir=summary_dir, log_dir=log_dir,
            baseline=densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3)))

    def set_trainable_layers(self):
        for layer in self.baseline.layers:
            # Posibilidad de poner tambien conv5_ directamente
            layer.trainable = ('conv5_block16' in layer.name) or ('conv5_block15' in layer.name) or \
                              ('conv5_block14' in layer.name)
        self.baseline.get_layer('bn').trainable = True
        self.baseline.get_layer('relu').trainable = True


