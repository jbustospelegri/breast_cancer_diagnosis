from utils.config import CLASS_LABELS
from utils.functions import create_dir, render_mpl_table

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from glob import glob
from typing import List
from collections import defaultdict

import os


class DataVisualizer:

    @staticmethod
    def get_dataframe_from_logs(dirname: str, test_name: str, metrics: list, *train_phases) -> pd.DataFrame:
        """

        Función utilizada para crear gráficas a partir de los historiales generados por keras durante el entrenamiento.

        :param dirname: directorio de almacenado del archivo de salida
        :param test_name: Nombre del test asignado durante el train. Esta variable servirá para filtrar los logs.
        :param metrics: lista de métricas a plotear. Estas métricas deben estar en los logs
        :param train_phases: Nombre(s) de la fase(s) presentes en el log de ejecución. Estas fases se identificarán a
                             partir del numero de épocas.
        :return: Dataframe con las metricas de cada modelo concatenadas.
        """

        # Lista para almacenar las métricas
        data_list = []

        # Se itera sobre los archivos almacenados en el directorio para almacenarlos en un dataframe. Estos archivos
        # se filtrarán mediante el nombre del test asignado durante el entrenamiento.
        for file in glob(os.path.join(f'{dirname}', f'*{test_name}.csv'), recursive=True):

            # Se lee el dataframe
            data = pd.read_csv(file, sep=';')

            # Se recupera el nombre del modelo a partir del nombre asignado al log.
            data.loc[:, 'Model'] = os.path.split(file)[1].split('_')[0]

            # Si se definen fases de entrenamiento se realiza un preprocesado de datos adicional.
            if len(train_phases) > 0:

                # Se crea una lista con las posiciones de aquellos registros cuya epoca sea 0 ya que indicarán el inicio
                # de una nueva fase de entrenamiento-
                pos = data[data.epoch == data.epoch.min()].index.tolist() + [len(data)]

                # Se cuentan las posiciones de diferencia entre cada cambio del tipo de entrenamiento para poder
                # reproducir las fases de enternamiento.
                n_elements = [pos[i + 1] - pos[i] for i in range(len(pos)-1)]

                # Se comprueba que existen las mismas posiciones de reinicio de epoca que de fases de entrenamiento
                # definidas por el usuario.
                assert len(n_elements) == len(train_phases), 'Número de fases de entrenamiento distinto al número de ' \
                                                             'reinicios de épocas detectadas'

                # Se crea una lista que reproduce el train_mode en función de las longitudes almacenadas en n_elements
                train_mode = []
                for n, name in zip(n_elements, train_phases):
                    train_mode.append([name] * n)

                # Se asigna la variable train mode al dataframe con las fases indicadas por el usuario
                data.loc[:, 'Train_Mode'] = [t for label in train_mode for t in label]
                # Se suma 1 a las épocas para empezar cada época en 1 y no en 0.
                data.loc[:, 'epoch'] = data.epoch + 1

            else:
                # En caso de no querer segregar por fases, se deja la variable train_mode vacía y se genera una
                # secuencia única a partir del índice con el objetivo de recrear las épocas de la fase de entrenamiento
                data.loc[:, 'Train_Mode'] = [''] * len(data)
                data.loc[:, 'epoch'] = data.index + 1

            # Se obtienen aquellas columnas que no contienen en el nombre alguna de las métricas definidas por el
            # usuario para evitar que se pivoten.
            melt_cols = [col for col in data.columns if not any(i in col for i in metrics)]

            # Se pivotan las métricas en una columna que se llamará Metrics_value. El nombre de cada métrica se
            # asignará a la columna Metrics_name.
            data_melt = data.melt(id_vars=melt_cols, var_name='Metrics_name', value_name='Metrics_value')

            # Se añade la columna modo a partir del nombre de la métrica. Por defecto, el callback de keras añade
            # la palabra val a cada una de las métricas, por lo que se utilizará dicha casuística para segregar en
            # entrenamiento y validación.
            data_melt.loc[:, 'Mode'] = np.where(data_melt.Metrics_name.str.contains('val', case=False), 'Val', 'Train')

            # Se añaden las métricas de cada modelo.
            data_list.append(data_melt)

        return pd.concat(data_list, ignore_index=True)

    @create_dir
    def plot_confusion_matrix(self, dirname: str, out_file: str, input_file: str):
        """
         Función que permite crear una matriz de confusión a partir de las predicciones generadas por el modelo en la
         fase de entrenamiento para graficarla en un archivo.

        :param dirname: directorio en el que se almacenará la imagen de salida.
        :param out_file: nombre del archivo de salida.
        :param input_file: nombre del archivo csv que contiene las predicciones.
        """

        # Lectura de los datos
        dataset = pd.read_csv(input_file, sep=';')

        # Se recuperan las columnas de cada modelo.
        models_name = [i for i in dataset.columns if i not in ['true_label', 'file', 'mode']]

        # En función del número de modelos, se generan n hileras para graficar los resultados. Cada hilera contendrá
        # dos modelos.
        nrows = (len(models_name) // 2) + 1

        # Se itera para cada fase de entrenamiento/validación de cada modelo.
        for mode in dataset['mode'].unique():

            # Se filtran los datos para obtener cada fase
            data = dataset[dataset['mode'] == mode]

            # Se crea la figura de matplotlib
            fig = plt.figure(figsize=(15, 10))

            # Se iteran los modelos
            for i, col in enumerate(models_name, 1):

                # Se crea un subplot.
                ax = fig.add_subplot(nrows, 2, i)

                # Se crea la tabla de contingencia a través de las clases verdaderas (columna true_label) y las
                # predecidas (columna definida por el nombre del modelo)..
                ct = pd.crosstab(data['true_label'], data[col], normalize=False)

                # Se muestra la matriz de confusión.
                sns.set(font_scale=1)
                sns.heatmap(ct.reindex(sorted(ct.columns), axis=1), cmap="Blues", annot=True, annot_kws={"size": 15},
                            fmt="d", cbar=False, ax=ax)

                # título y eje x del gráfico
                ax.set_title(f'Modelo: {col}\n{mode}', fontweight='bold', size=14)
                ax.set(xlabel='Predictions')

            # Se ajustan los subplots
            fig.tight_layout()
            # Se almacena la figura.
            fig.savefig(os.path.join(dirname, f'{out_file}_{mode}.jpg'))

    @staticmethod
    def get_metrics_matrix(input_file: str, dirname: str, out_file: str, class_metrics: bool = False):
        """
        Función utilizada para generar una imagen con una tabla que contiene las métricas de accuracy, precision,
        recall y f1_score para entrenamiento y validación. Las métricas se calculan a partir del log de predicciones
        generado por un modelo.
        :param input_file: path del log de predicciones
        :param dirname: carpeta en la que se guardarán las imagenes generadas
        :param out_file: nombre del archivo de imagen a generar
        :param class_metrics: booleano que sirve para calcular las métricas de precision, recall, accuracy y f1 de cada
                              clase y modelo en caso de ser true. En caso contrario, se generarán las métricas de forma
                              global para cada modelo.
        """

        # Lectura del set de datos
        dataset = pd.read_csv(input_file, sep=';')

        # Se recuperan las columnas referentes a métricas de modelos,
        models_name = [i for i in dataset.columns if i not in ['true_label', 'file', 'mode']]

        # Se crea un dataset que contendrá las métricas accuracy, precision, recall y f1 para train y validación a nivel
        # general. En caso de tener el parametro class_metrics a True, las columnas del dataset serán un multiindice con
        # modelos y clases; en caso contrario, únicamente contendrá el nombre de los modelos.
        metrics = pd.DataFrame(
            index=pd.MultiIndex.from_product([['train', 'val'], ['Accuracy', 'Precision', 'Recall', 'F1_score']]),
            columns=pd.MultiIndex.from_product([models_name, dataset.true_label.unique().tolist()]) if class_metrics
            else models_name
        )

        # Se asigna el nombre del índice
        metrics.index.set_names(['mode', 'metric'], inplace=True)

        # Se itera sobre cada modelo
        for model in models_name:
            # Se iteran los datos de entrenamiento y validación
            for mode in ['train', 'val']:

                # Filtrado de datos de interes
                df = dataset[dataset['mode'] == mode]

                if class_metrics:

                    # En caso de querer obtener las metricas de cada clase se itera sobre cada una de estas.
                    for _, class_label in CLASS_LABELS.items():

                        # Para poder generar las métricas deseadas, se considerará cada clase como verdadera asignandole
                        # el valor 1, y el resto de clases con el valor 0. De esta forma, se evaluará para cada clase.
                        # (Técnica one vs all)
                        map_dict = defaultdict(lambda: 0, {class_label: 1})
                        # Creación del dataset de métricas
                        metrics.loc[(mode,), (model, class_label)] = [
                            round(accuracy_score(df.true_label.map(map_dict), df[model].map(map_dict)) * 100, 2),
                            round(precision_score(df.true_label.map(map_dict), df[model].map(map_dict), zero_division=0,
                                                  average='weighted') * 100, 2),
                            round(recall_score(df.true_label.map(map_dict), df[model].map(map_dict), zero_division=0,
                                               average='weighted') * 100, 2),
                            round(f1_score(df.true_label.map(map_dict), df[model].map(map_dict), zero_division=0,
                                           average='weighted') * 100, 2)
                        ]

                else:
                    # Creación del dataset de métricas
                    metrics.loc[(mode, ), model] = [
                        round(accuracy_score(df.true_label.tolist(), df[model].tolist()) * 100, 2),
                        round(precision_score(df.true_label.tolist(), df[model].tolist(), zero_division=0,
                                              average='weighted') * 100, 2),
                        round(recall_score(df.true_label.tolist(), df[model].tolist(), zero_division=0,
                                           average='weighted') * 100, 2),
                        round(f1_score(df.true_label.tolist(), df[model].tolist(), zero_division=0,
                                       average='weighted') * 100, 2)
                    ]

        # se resetea el índice para poder mostrar en la tabla si las métricas son de entrenamiento o de validación
        metrics.reset_index(inplace=True)
        # se crea la tabla en formato de imagen y se almacena.
        fig, _ = render_mpl_table(metrics)
        fig.savefig(os.path.join(dirname, f'{out_file}.jpg'))

    @staticmethod
    def plot_model_metrics(plot_params: List[dict], dirname: str = None, filename: str = None, plots_per_line: int = 2):
        """
        Función para representar gráficamente las métricas obtenidas especificadas mediante el parámetro plot_params
        :param plot_params: lista de diccionarios que contiene las especificaciones para generar un grafico
        :param dirname: nombre del directorio en el que se guardará la imagen
        :param filename: nombre del archivo con el que se guardará la imagen
        :param plots_per_line: determina el número de columnas de cada figura
        """

        # Se crea la figura y los subplots
        nrows = (len(plot_params) // plots_per_line) + 1
        figure = plt.Figure(figsize=(15, 10))

        # Se itera cada diccionario almacenado en plot params
        for i, plot_configuration in enumerate(plot_params, 1):

            # Se crea el subplot
            ax = figure.add_subplot(nrows, plots_per_line, i)

            assert 'plot' in plot_configuration, 'debe especificar el diccionario data en plot_params'

            # Se crea un gráfico lineal con las especificiaciones de plot
            sns.lineplot(**plot_configuration['plot'], ax=ax)

            # Configuración de los labels del gráfico
            if plot_configuration.get('labels', False):
                ax.set(xlabel=plot_configuration['labels'].get('xlabel', plot_configuration['plot']['x']),
                       ylabel=plot_configuration['labels'].get('ylabel',  plot_configuration['plot']['y']))

            # Configuración del título del gráfico
            if plot_configuration.get('title', False):
                ax.title.set(**plot_configuration['title'])

            # Configuración de la leyenda del gráfico
            if plot_configuration.get('legend', False):
                ax.legend(**plot_configuration['legend'])

            # Se elimina el grid del grafico
            ax.grid(False)

        # Ajuste de la figura y guardado de la imagen
        figure.tight_layout()
        figure.savefig(os.path.join(dirname, filename))

    @create_dir
    def get_model_logs_metrics(self, logs_dir: str, test_name: str, dirname: str, out_filename: str,
                               train_phases: list = None):
        """
        Función para generar las métricas de accuracy, f1 score y loss a partir de los logs generados durante la fase
        de entrenamiento de un modelo
        :param logs_dir: directorio en el cual se almacenan los logs
        :param test_name: nombre de la prueba asignada a la fase de entrenamiento
        :param dirname: directorio para almacenar las gráficas
        :param out_filename: nombre del archivo para almacenar las gráficas
        :param train_phases: nombre de las fases de entrenamiento con las que se ha entrenado un modelo
        """

        if train_phases is None:
            train_phases = []

        # Se recupera un dataframe a partir del directorio de logs que contendrá las métricas
        data = self.get_dataframe_from_logs(logs_dir, test_name, ['accuracy', 'f1_score', 'loss'], *train_phases)

        # Se itera sobre las fases con las que se ha entrenado cada modelo
        for train_mode in data.Train_Mode.unique():
            # Filtrado de datos
            data_filtered = data[data.Train_Mode == train_mode]

            # Se crea la gráfica correspondiente
            self.plot_model_metrics(
                dirname=dirname, filename=f'{out_filename}{"_" + train_mode if train_mode else ""}.jpg',
                plot_params=[
                    {
                        'plot': {
                            'data': data_filtered[
                                data_filtered['Metrics_name'].str.contains(metric, case=False, na=False)],
                            'y': 'Metrics_value',
                            'x': 'epoch',
                            'hue': 'Model',
                            'style': 'Mode',
                        },
                        'labels': {
                            'xlabel': 'Epocas',
                            'ylabel': ylabel,
                        },
                        'title': {
                            'text': f'{title} {train_mode}',
                            'fontsize': 12,
                            'fontweight': 'bold'
                        },
                        'legend': {
                            'fontsize': 'x-small',
                            'frameon': False,
                            'framealpha': 1
                        }
                    } for title, metric, ylabel, in zip(['Exactitud Modelos', 'Pérdidas Modelos', 'F1_Score Modelos'],
                                                        ['accuracy', 'loss', 'f1_score'],
                                                        ['Exactitud', 'Pérdidas', 'F1'])
                ]
            )
