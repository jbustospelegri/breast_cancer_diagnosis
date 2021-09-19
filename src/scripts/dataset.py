from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img, DataFrameIterator
from sklearn.model_selection import train_test_split

from utils.functions import create_dir, render_mpl_table
from utils.config import SEED, PROCESED_DATA_PATH

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import glob
import cv2
import os
import logging


# Definición de estilos de la librería seaborn
sns.set_style(style='whitegrid')
sns.set_color_codes('pastel')

# Modificación ploteado pandas
pd.options.display.float_format = "{:,.2f}".format


class Dataset:
    df = None
    clases = {}
    n_clases = 0

    def __init__(self, path: str, image_format: str = 'jpg'):
        self.path = path
        self.image_format = image_format

    def __repr__(self):
        return f'Dataset formado por {len(self.df)} registros.\nExisten {len(self.df["class"].unique())} clases\n' + \
               f'Distribución de las clases en train / val\n\n' + \
               f"{self.df.groupby('dataset')['class'].value_counts(normalize=True).unstack()}"

    def get_dataset(self):
        """
        Función que genera un dataframe con las imagenes de cada clase mediante la busqueda de los subdirectorios
        indicados en self.path y cuyo formato de imagen sea el indicado en self.image_format. La estructurá debe ser:
            path/
            ...class_a/
            ......a_image_1.jpg
            ......a_image_2.jpg
            ...class_b/
            ......b_image_1.jpg
            ......b_image_2.jpg

        El dataset generado estará formado por 3 columnas; una para la clase de la imagen ('class'), otra para su
        path (item_path) y otra para identificar a qué conjunto pertenece ('dataset') cuyos valores serán train o val..
        """

        elements = []

        print('-' * 50 + f'\nLeyendo directorio de imagenes {self.path} para generar datataset')

        # Se itera sobre el directorio de imagenes obteniendo cada imagen
        for file in glob.glob(os.path.join(self.path, '*', f'*{self.image_format}'), recursive=True):
            # Se obtiene el nombre de la clase representado mediante el nombre de la carpeta
            label = os.path.basename(os.path.dirname(file))

            # Se llena el dataframe con el nombre de clase, número de observaciones y 5 ejemplos en formato de lista.
            elements.append([label, file, 'train'])

        # Dataframe que almacenará el nombre de clase, número de imagenes y un ejemplo de cada categoria.
        self.df = pd.DataFrame(columns=['class', 'item_path', 'dataset'], data=elements)

        # Se almacena el número de clases
        self.n_clases = len(self.df['class'].unique())

        print(f'Dataset Leido correctamente - {len(self.df)} Imagenes encontradas')

    @create_dir
    def get_class_distribution_from_dir(self, dirname: str, filename: str):
        """
        Función que permite representar graficamente el número de observaciones y la proporción de cada una de las
        clases presentes en un dataet. La clase de cada observción debe estar almacenada en una columna cuyo
        nombre sea "class".

        :param dirname: directorio en el que se almacenará la imagen.
        :param filename: nombre de la imagen.
        """

        # Se obtiene el dataset en caso de no existir.
        if self.df is None:
            self.get_dataset()

        print('-' * 50 + f'\nObteniendo distribución de clases del dataset. - {len(self.df["class"].unique())} clases '
                         f'encotradas')

        # Figura de matplotlib para almacenar el gráfico
        plt.figure(figsize=(15, 5))

        # Gráfico de frecuencias
        ax = sns.countplot(x=self.df['class'], palette=sns.light_palette((210, 90, 60), input='husl', reverse=True))

        # Se realizan las anotaciones de los valores de frecuencia y frecuencia normalizda para cada valor de la
        # variable objetivo.
        for p in ax.patches:
            ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x() + p.get_width() * 0.2, p.get_height() + 10))
            ax.annotate('({:.2f}%)'.format((p.get_height() / len(self.df)) * 100),
                        (p.get_x() + p.get_width() * 0.45, p.get_height() + 10))

        # Título del gráfico
        ax.set_title('Proporción atributos/clase', fontweight='bold', size=14)

        # Se elimina el label del eje y.
        ax.set(ylabel='')

        # Se almacena la figura
        plt.savefig(os.path.join(dirname, filename))

        print(f'Gráfico de distribución de clases almacenada en {os.path.join(dirname, filename)}')

    @create_dir
    def get_class_examples(self, dirname: str, filename: str, n_example: int = 5):
        """
        Función que permite representar graficamente un conjunto de imagenes de cada clase presente en el dataset.
        Dichas imagenes se almacenarán en forma de grid en un archivo de imagen.

        :param filename: nombre con el que se guardará la imagen.
        :param dirname: directorio en el que se guardará la imagen.
        :param n_example: número de imagenes por clase.
        """

        print('-' * 50 + f'\nObteniendo {n_example} imagenes de ejemplo por clase')

        # Se recupera el dataset en caso de no existir.
        if self.df is None:
            self.get_dataset()

        # Se crea la figura de matplotlib formada por una cuadricula de subplots. En cada hilera de la cuadrícula se
        # representará una clase y, en cada columna, una imagen de ejemplo distinta.
        fig, ax = plt.subplots(self.n_clases, n_example, figsize=(15, 10))

        # Se iteran las distintas clases del dataset recuperando los top n muestras de ejemplo.
        for axi, image in zip(ax.flat, self.df.groupby('class').item_path.head(n_example)):
            # Se representa la imagen mediante imshow y se lee mediante imread de opencv2. Por defecto imread de cv2
            # lee las imagenes en formato BGR, por lo que deberán de transformarse en formato RGB mediante cvtColor.
            axi.imshow(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))

            # Se asigna la clase de cada imagen en el eje x de cada gráfico. La clase, se obtendrá a partir del
            # directorio de cada imagen.
            axi.set(xticks=[], yticks=[], xlabel=os.path.basename(os.path.dirname(image)))

        # Ajuste de la figura.
        fig.tight_layout()

        # Se almacena la figura
        plt.savefig(os.path.join(dirname, filename))
        print(f'Ejemplos almacenados en {os.path.join(dirname, filename)}')

    @create_dir
    def get_info_from_image(self, dirname: str, log_file: str):
        """
        Función que realiza un análisis descriptivo de los distintos tamaños de imagen presentes en el dataset y del
        rango dinámico de estas. Este análisis descriptivo se almacena en formato de tabla en un archivo jpg o png.

        :param dirname: directorio para almacenar la imagen
        :param log_file: nombre del archivo a crear.
        """

        # Se recupera el dataset en caso de no existir.
        if self.df is None:
            self.get_dataset()

        # Se crea un dataframe sobre el que se almacenará el análisis descriptivo de la altura, anchura y rango dinámico
        # de las imagenes.
        analysis_df = pd.DataFrame(columns=['Dynamic Range', 'Image Widht', 'Image Height'])

        # Se itera cada una de las imagenes presentes en el dataset (representadas mediante la columna item_path).
        for filepath in self.df.item_path.unique():
            # Se lee cada una de las imagenes y se convierte a escala de grises
            imag = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2GRAY)

            # Se almacena para cada imagen su rango dinámico representado mediante la diferencia entre el píxel más alto
            # y más bajo; la anchura y la altura de cada imagen.
            analysis_df.loc[len(analysis_df), :] = [imag.max() - imag.min(), imag.shape[0], imag.shape[1]]

        # Se representa la tabla mediante un table de matplotlib
        fig, _ = render_mpl_table(analysis_df.astype(int).describe().round(2).reset_index().
                                  rename(columns={'index': ''}))
        # Se almacena la figura
        fig.savefig(os.path.join(dirname, log_file))

    def split_dataset_on_train_val(self, train_prop: float, stratify: bool = False):
        """
        Función que permite dividir el dataset en un subconjunto de entrenamiento y otro de validación. La división
        puede ser estratificada. Dicha división se realizará mediante la columna 'dataset' del set de datos de self.df,
        asignadno el valor 'train' para el conjunto de entrenamiento y 'val' para el conjunto de validacion.

        :param train_prop: proporción del total de observaciones del dataset con los que se creará el conjunto de train
        :param stratify: booleano que determina si la división debe ser estratificada o no.
        """

        # Se confirma que el valor de train_prop no sea superior a 1
        assert train_prop < 1, 'Proporción de datos de validación superior al 100%'

        # Se recupera el dataset en caso de no existir
        if self.df is None:
            self.get_dataset()

        # Se realiza la división del conjunto de datos en train y validación. Únicamente interesa recuperar el conjunto
        # de entrenamiento ya que los paths se utilizarán posteriormente para determinar la columna 'dataset' de self.df
        train_x, _, _, _ = train_test_split(self.df.item_path, self.df['class'], random_state=SEED,
                                            stratify=self.df['class'] if stratify else None)

        # Se asigna el valor de 'train' a aquellas imagenes (representadas por sus paths) que estén presentes en train_x
        # en caso contrario, se asignará el valor 'val'.
        self.df.dataset = np.where(self.df.item_path.isin(train_x), 'train', 'val')

    def get_dataset_generator(self, batch_size: int, size: tuple = (224, 224), directory: bool = False,
                              preprocessing_function=None):
        """

        Función que permite recuperar un dataframe iterator para entrenamiento y para validación.

        :param directory: booleano que permite almacenar las imagenes generadas por el ImageDataGenerator en la carpeta
                          Data/01_procesed.
        :param batch_size: tamaño de batch con el que se crearán los iteradores.
        :param size: tamaño de la imagen que servirá de input para los iteradores. Si la imagen tiene un tamaño distinto
                     se aplicará un resize aplicando la tecnica de interpolación lanzcos. Por defecto es 224, 224.
        :param preprocessing_function: función de preprocesado a aplicar a las imagenes leidas una vez aplicadas las
                                       técnicas de data augmentation.
        :return: dataframeIterator de validación y de tran.
        """

        # Se crea una configuración por defecto para crear los dataframe iterators. En esta, se leerán los paths de las
        # imagenes a partir de una columna llamada 'item_path' y la clase de cada imagen estará representada por la
        # columna 'class'. Para ajustar el tamaño de la imagen al tamaño definido por el usuario mediante input, se
        # utilizará la técnica de interpolación lanzcos. Por otra parte, para generar una salida one hot encoding
        # en función de la clase de cada muestra, se parametriza class_mode como 'categorical'.
        params = dict(
            x_col='item_path',
            y_col='class',
            target_size=size,
            interpolation='lanczos',
            shufle=True,
            seed=SEED,
            batch_size=batch_size,
            class_mode='categorical',
            directory=PROCESED_DATA_PATH if directory else None
        )

        # Parametrización del generador de entrenamiento. Las imagenes de entrenamiento recibirán un conjunto de
        # modificaciones aleatorias con el objetivo de aumentar el set de datos de entrenamiento y evitar de esta forma
        # el over fitting. Entre ellas, destacan giros aleatorios tanto en el eje vertical y horizontal; zoom de como
        # máximo un 10% y deformaciónes en el eje vertical e horizontal de como máximo 10 grados. Por otra parte, se
        # les aplicará la técnica de preprocesado subministrada por el usuario.
        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.1,
            zoom_range=0.1,
            preprocessing_function=preprocessing_function
        )

        # Parametrización del generador de validación. Las imagenes de validación exclusivamente se les aplicará la
        # técnica de preprocesado subministrada por el usuario.
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function
        )

        # Para evitar entrecruzamientos de imagenes entre train y validación a partir del atributo shuffle=True, cada
        # generador se aplicará sobre una muestra disjunta del set de datos representada mediante la columna dataset.

        # Se chequea que existen observaciones de entrenamiento para poder crear el dataframeiterator.
        if len(self.df[self.df.dataset == 'train']) == 0:
            train_df_iter = None
            logging.warning('No existen registros para generar un generador de train. Se retornará None')
        else:
            train_df_iter = train_datagen.flow_from_dataframe(dataframe=self.df[self.df.dataset == 'train'], **params)

        # Se chequea que existen observaciones de validación para poder crear el dataframeiterator.
        if len(self.df[self.df.dataset == 'val']) == 0:
            val_df_iter = None
            logging.warning('No existen registros para generar un generador de validación. Se retornará None')
        else:
            val_df_iter = val_datagen.flow_from_dataframe(dataframe=self.df[self.df.dataset == 'val'], **params)

        return train_df_iter, val_df_iter

    @staticmethod
    def get_data_augmentation_examples(dirname: str, out_file: str, example_imag: str):
        """
        Función que permite generar un ejemplo de cada tipo de data augmentation aplicado

        :param dirname: directorio de guardado
        :param out_file: nombre del archivo de imagen a generar
        :param example_imag: nombre de una muestra de ejemplo sobre la que se aplicarán las transformaciones propias del
                             data augmentation
        """

        def plot_image(img, title, ax_):
            """
            Función que permite representar una imagen en un axes de matplotlib suprimiendole el grid y los ejes.

            :param img: imagen en formato array y de dimensiones (n, width, height, channels)
            :param title: título del axes
            :param ax_: axes subplot

            """

            # Se representa la imagen
            ax_.imshow(array_to_img(img[0]))

            # Se eliminan ejes y grid
            ax_.axes.grid(False)
            ax_.axes.set_xticklabels([])
            ax_.axes.set_yticklabels([])

            # Título del gráfico en el eje de las x.
            ax_.axes.set(xlabel=title)

        # Se lee la imagen del path de ejemplo
        image = load_img(example_imag)
        # Se transforma la imagen a formato array
        image = img_to_array(image)
        # Se añade una dimensión para obtener el dato de forma (1, width, height, channels)
        image_ori = np.expand_dims(image, axis=0)

        # Figura y subplots de matplotlib. Debido a que existen 4 transformaciones de data augmentation, se creará un
        # grid con 5 columnas que contendrán cada ejemplo de transformación y la imagen original
        fig, axes = plt.subplots(1, 5, figsize=(15, 7))

        # Se representa la imagen original en el primer subplot.
        plot_image(img=image_ori, title='Imagen Original', ax_=axes[0])

        # Se iteran las transformaciones
        for transform_title, parameters, ax in zip(['horizontal_flip', 'vertical_flip', 'shear_range', 'zoom_range'],
                                                   [True, True, 0.1, 0.1], axes[1:].flatten()):

            # Se crea al datagenerator con exclusivamente la transformación a aplicar.
            datagen = ImageDataGenerator(**{transform_title: parameters})
            # Se recupera la imagen transformada mediante next() del método flow del objeto datagen
            plot_image(img=next(datagen.flow(image_ori)), title=transform_title, ax_=ax)

        # Se almacena la figura
        plt.savefig(os.path.join(dirname, out_file))


class TestDataset(Dataset):

    def __init__(self, path: str, image_format: str = 'jpg'):
        super().__init__(path=path, image_format=image_format)

    def __repr__(self):
        return f'Dataset formado por {len(self.df)} registros.'

    def get_dataset(self):
        """

        Función que genera un dataframe con las imagenes almacenadas en la carpeta definida en self.path.
        Este dataframe estará formado por tres columnas: 'item_path' con el directorio de cada imagen; 'class': no
        contendrá información pero se incluye por temas de herencia; 'dataset': contendrá el valor fijo 'Test'.

        """

        elements = []

        print('-' * 50 + f'\nLeyendo directorio de imagenes {self.path} para generar datataset')

        # Se itera sobre el directorio de imagenes obteniendo cada imagen
        for file in glob.glob(os.path.join(self.path, f'*{self.image_format}'), recursive=True):

            elements.append([file, np.nan, 'Test'])

        # Dataframe que almacenará el nombre de clase, número de imagenes y un ejemplo de cada categoria.
        self.df = pd.DataFrame(columns=['item_path', 'class', 'dataset'], data=elements)

        print(f'Dataset Leido correctamente - {len(self.df)} Imagenes encontradas para test')

    def get_dataset_generator(self, batch_size: int, size: tuple = (224, 224), directory: bool = False,
                              preprocessing_function=None) -> DataFrameIterator:
        """

        Función que permite recuperar un dataframe iterator para test.

        :param directory: booleano que permite almacenar las imagenes generadas por el ImageDataGenerator en la carpeta
                          Data/01_procesed.
        :param batch_size: tamaño de batch con el que se crearán los iteradores.
        :param size: tamaño de la imagen que servirá de input para los iteradores. Si la imagen tiene un tamaño distinto
                     se aplicará un resize aplicando la tecnica de interpolación lanzcos. Por defecto es 224, 224.
        :param preprocessing_function: función de preprocesado a aplicar a las imagenes leidas una vez aplicadas las
                                       técnicas de data augmentation.
        :return: dataframeIterator de test.
        """

        # Se comprueba que existe el dataframe en self.df
        assert self.df is not None, 'Dataframe vacío. Ejecute la función get_dataset()'

        # Se crea un datagenerator en el cual únicamente se aplicará la función de preprocesado introducida por el
        # usuario.
        datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function
        )

        # Se devuelve un dataframe iterator con la misma configuración que la clase padre Dataset a excepción de que en
        # este caso, la variable class_mode es None debido a que el dataframe iterator deberá contener exclusivamente
        # información de las observaciones y no sus clases.
        return datagen.flow_from_dataframe(
            dataframe=self.df,
            x_col='item_path',
            y_col='class',
            target_size=size,
            interpolation='lanczos',
            shufle=False,
            seed=SEED,
            batch_size=batch_size,
            class_mode=None,
            directory=PROCESED_DATA_PATH if directory else None
        )

    def split_dataset_on_train_val(self, train_prop: float, stratify: bool = False):
        # Se inhabilita la función debido a que la clase TestDataset sólo contendrá información de observaciones de Test
        logging.warning('Función no válida para la clase TestDataset')
