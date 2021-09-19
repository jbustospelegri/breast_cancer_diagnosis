import keras
import pickle
import os

from typing import Callable
from keras import Model, Sequential
from time import time

from src.utils.functions import create_dir, log_cmd_to_file, f1_score
from src.utils.config import LOGGING_DATA_PATH, MODEL_SUMMARY_DATA_PATH, SEED

from keras.preprocessing.image import DataFrameIterator
from keras.callbacks import EarlyStopping, CSVLogger, History
from keras.optimizers.adam_v2 import Adam
from keras.api.keras.applications import resnet50, densenet, vgg16, inception_v3
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D

from sklearn.ensemble import GradientBoostingClassifier

import numpy as np
import pandas as pd


class GeneralModel:

    model: Model = None
    optimizer: keras.optimizers = None
    callbakcs = []
    history: History = None

    def __init__(self, n_clases: int, name: str = 'baseline', baseline: keras.Model = None, test_name: str = None,
                 input_shape: tuple = (250, 250), preprocess_func: Callable = None, get_model_structure: bool = False,
                 log_dir: str = LOGGING_DATA_PATH,  summary_dir: str = MODEL_SUMMARY_DATA_PATH):
        self.name = name
        self.baseline = baseline
        self.n_clases = n_clases
        self.input_shape = input_shape
        self.test_name = test_name
        self.preprocess_func = preprocess_func
        self.log_dir = log_dir
        self.log_filename = self.get_dir(dirname=log_dir, file_ext='csv', file_args=[self.name, self.test_name])
        self.summary_dir = summary_dir
        self.summary_filename = self.get_dir(dirname=summary_dir, file_ext='txt', file_args=[self.name, self.test_name])
        self.create_model(print_model=get_model_structure)

    @create_dir
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
        out = Dense(512, activation='relu')(out)
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

        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1_score])

    @create_dir
    def get_dir(self, dirname: str, file_ext: str, file_args: list = None):
        """
        Función que devuelve un filepath del modelo y crea la carpeta del filepath en caso de no existir
        :param dirname: nombre del directorio
        :param file_ext: extensión del archivo rchivo
        :param file_args: nombres adicionales a añadir al nombre del archivo creado
        :return: filepath
        """

        if file_args is None:
            file_args = [f'default_{self.name}']

        return os.path.join(dirname, f'{"_".join(list(filter(None, file_args)))}.{file_ext}')

    def set_model_callbacks(self, early_stopping: bool = True, csv_logger: bool = True):
        """
        Función utilizada para añadir callbacks a la fase de entrenamiento de un modelo
        :param early_stopping: booleano que permite añadir un callback de early stopping
        :param csv_logger:  boleano que permite añadir un callback de csvlogger
        """

        self.callbakcs = []

        if early_stopping:
            self.callbakcs.append(EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True))

        if csv_logger:
            self.callbakcs.append(CSVLogger(filename=self.log_filename, separator=';', append=True))

    def set_trainable_layers(self):
        """
        Función utilizada para setear layers del modelo como entrenables. Este metodo será sobreescrito por las clases
        heredadas
        """
        pass

    def train_model(self, epochs: int, batch: int, train_data: DataFrameIterator, val_data: DataFrameIterator = None):
        """
        Función utilizada para entrenar un modelo
        :param epochs: número de epocas
        :param batch: tamaño de batch
        :param train_data: dataframe iterator con los datos de entrenamiento
        :param val_data: dataframe iterator con los datos de validación
        """

        self.history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            verbose=2,
            callbacks=self.callbakcs,
            steps_per_epoch=train_data.samples // batch,
            validation_steps=val_data.samples // batch,
        )

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

        self.train_model(epochs=epochs, batch=batch_size, train_data=train_data, val_data=val_data)

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

    @create_dir
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

    def __init__(self, name: str, n_clases: int, test_name: str = '', get_model_structure: bool = False):
        super().__init__(n_clases=n_clases, name=name, preprocess_func=vgg16.preprocess_input, input_shape=(224, 224),
                         test_name=test_name, get_model_structure=get_model_structure,
                         baseline=vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3)))

    def set_trainable_layers(self):
        for layer in self.baseline.layers:
            layer.trainable = 'block5' in layer.name


class Resnet50Model(GeneralModel):

    def __init__(self, name: str, n_clases: int, test_name: str = '', get_model_structure: bool = False):
        super().__init__(n_clases=n_clases, name=name, preprocess_func=resnet50.preprocess_input,
                         test_name=test_name, get_model_structure=get_model_structure, input_shape=(224, 224),
                         baseline=resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3)))

    def set_trainable_layers(self):
        for layer in self.baseline.layers:
            layer.trainable = ('conv5_block3_' in layer.name) or ('conv5_block2_' in layer.name)


class InceptionV3Model(GeneralModel):

    def __init__(self, name: str, n_clases: int, test_name: str = '', get_model_structure: bool = False):
        super().__init__(
            n_clases=n_clases, name=name, preprocess_func=inception_v3.preprocess_input, input_shape=(299, 299),
            test_name=test_name, get_model_structure=get_model_structure,
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

    def __init__(self, name: str, n_clases: int, test_name: str = '', get_model_structure: bool = False):
        super().__init__(
            n_clases=n_clases, name=name, preprocess_func=densenet.preprocess_input, input_shape=(224, 224),
            test_name=test_name, get_model_structure=get_model_structure,
            baseline=densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3)))

    def set_trainable_layers(self):
        for layer in self.baseline.layers:
            # Posibilidad de poner tambien conv5_ directamente
            layer.trainable = ('conv5_block16' in layer.name) or ('conv5_block15' in layer.name) or \
                              ('conv5_block14' in layer.name)
        self.baseline.get_layer('bn').trainable = True
        self.baseline.get_layer('relu').trainable = True


class ModelEnsamble:

    def __init__(self, model_path: str = ''):
        if os.path.exists(model_path):
            self.model_gb = pickle.load(open(model_path, 'rb'))
        else:
            self.model_gb = GradientBoostingClassifier(max_depth=3, n_estimators=20, random_state=SEED)

    @staticmethod
    def get_dataframe_from_models(data: pd.DataFrame, **model_inputs):
        """
        Función utilizada para generar un set de datos unificado a partir de las predicciones generadas por cada modelo
        :param data: dataframe con el nombre de archivo
        :param model_inputs: kwargs cuya key será el nombre del modelo que genera las predicciones y cuyo value será
                             un dataframe formado por las predicciones (columna predictions) y el filename de cada
                             observación.
        :return: dataframe unificado
        """
        for model, df in model_inputs.items():
            data = pd.merge(
                left=data,
                right=df.set_index('filename').rename(columns={'predictions': model})[[model]],
                right_index=True,
                left_index=True,
                how='left'
            )
        return data

    @create_dir
    def train_model(self, train: pd.DataFrame,  dirname: str,  filename: str, val: pd.DataFrame = None,
                    save_model: str = '', **models_ensamble):
        """
        Función utilizada para generar el algorítmo de gradient boosting a partir de las predicciones generadas por cada
        modelo.
        :param train: Dataframe que contendrá los identificadores (filenames) del conjunto del set de entrenamiento y
                      una columna 'mode' cuyo valore sera train. Además deberá contener la columna true_label con las
                      clases verdaderas de cada observación
        :param dirname: directorio en el que se almacenarán las predicciones del modelo
        :param filename: nombre con el que se almacenarán las predicciones del modelo
        :param val:  Dataframe que contendrá los identificadores (filenames) del conjunto del set de validación y
                     una columna 'mode' cuyo valor sera val. Además deberá contener la columna true_label con las
                      clases verdaderas de cada observación
        :param save_model: nombre del archivo con el que se guardará el modelo.
        :param models_ensamble: kwargs que contendrá como key el nombre del modelo y como values los valores devueltos
                                por el método predict de cada modelo
        """

        # En caso de existir dataset de validación, se concatena train y val en un dataset único. En caso contrario,
        # se recupera unicamente el set de datos de train
        if val is not None:
            gb_dataset = pd.concat(objs=[train.set_index('filenames'), val.set_index('filenames')], ignore_index=False)
        else:
            gb_dataset = train

        # se asigna el nombre del indice
        gb_dataset.index.name = 'file'

        # Se unifica el set de datos obteniendo las predicciones de cada modelo representadas por models ensamble
        df = self.get_dataframe_from_models(gb_dataset, **models_ensamble)

        # generación del conjunto de datos de train para gradient boosting
        train_gb_x = df[df['mode'].str.lower() == 'train'][list(models_ensamble.keys())]
        train_gb_y = df[df['mode'].str.lower() == 'train'][['true_label']]

        # generación del conjunto de datos de validación
        if val is not None:
            val_gb_x = df[df['mode'].str.lower() == 'val'][list(models_ensamble.keys())]

        # entrenamiento del modelo
        self.model_gb.fit(pd.get_dummies(train_gb_x), np.reshape(train_gb_y.values, -1))

        # se añade al dataset original, las predicciones del modelo de gradient boosting
        df.loc[:, 'Gradient_Boosting'] = [*self.model_gb.predict(pd.get_dummies(train_gb_x)),
                                          *self.model_gb.predict(pd.get_dummies(val_gb_x))]

        # se almacenan las predicciones
        savefile = os.path.join(dirname, filename)
        df.reset_index().to_csv(savefile, sep=';', index=False)

        # se almacena el modelo en caso de que el usuario haya definido un nombre de archivo
        if save_model:
            pickle.dump(self.model_gb, open(save_model, 'wb'))

    @create_dir
    def predict(self, dirname: str, filename: str, data: DataFrameIterator, return_model_predictions: bool = False,
                **input_models):
        """
        Función utilizada para realizar la predicción del algorítmo de graadient boosting a partir de las predicciones
        del conjunto de redes convolucionales
        
        :param dirname: directorio en el que se almacenará el log de predicciones
        :param filename: nombre del archivo en el que se almacenará el log de predicciones
        :param data: dataframe que contiene el nombre de cada imagen en una columna llamada filenames
        :param return_model_predictions: booleano que permite recuperar en el log de predicciones, las predicciones
                                         individuales de cada red neuronal convolucional
        :param input_models: kwargs que contendrá como key el nombre del modelo y como values los valores devueltos
                             por el método predict de cada modelo de red neuronal convolucional
        """

        # Se genera un dataframe con los directorios de las imagenes a predecir
        gb_dataset = pd.DataFrame(index=data.filenames)
        gb_dataset.index.name = 'image'

        # Se unifica el set de datos obteniendo las predicciones de cada modelo representadas por input_models
        df = self.get_dataframe_from_models(gb_dataset, **input_models)

        # Se añaden las predicciones
        df.loc[:, 'label'] = self.model_gb.predict(pd.get_dummies(df[input_models.keys()]))

        # se escribe el log de errores con las predicciones individuales de cada arquitectura de red o únicamente las
        # generadas por gradient boosting
        if return_model_predictions:
            df.to_csv(os.path.join(dirname, filename), sep=';')
        else:
            df[['label']].to_csv(os.path.join(dirname, filename), sep=';')
