import io
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from itertools import product
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, roc_auc_score, roc_curve
from typing import List
from collections import defaultdict
from albumentations import Compose

from algorithms.utils import optimize_threshold
from data_viz.functions import render_mpl_table, plot_image, create_countplot
from preprocessing.image_processing import full_image_pipeline, crop_image_pipeline
from utils.config import (
    MODEL_FILES, ENSEMBLER_CONFIG, CLASSIFICATION_METRICS, CLASSIFICATION_DATA_AUGMENTATION_FUNCS
)
from utils.functions import get_path, search_files, get_filename
from algorithms.utils import apply_bootstrap

sns.set(style='white')
sns.despine()


class DataVisualizer:
    """
        Clase para realizar las visualizaciones de los modelos de la red
    """

    metrics = [f.lower() if type(f) is str else f.__name__ for f in CLASSIFICATION_METRICS.values()] + \
              ['accuracy', 'loss']
    labels = {
        'BENIGN': 0,
        'MALIGNANT': 1
    }

    def __init__(self, config: MODEL_FILES, img_type: str):
        self.conf = config
        self.img_type = img_type

    @staticmethod
    def get_dataframe_from_logs(dirname: io, metrics: list) -> pd.DataFrame:
        """

        Función utilizada para crear gráficas a partir de los historiales generados por keras durante el entrenamiento.

        :param dirname: directorio de almacenado del archivo de salida
        :param metrics: lista de métricas a plotear. Estas métricas deben estar en los logs
        """

        # Lista para almacenar las métricas
        data_list = []

        # Se itera sobre los archivos almacenados en el directorio para almacenarlos en un dataframe. Estos archivos
        # se filtrarán mediante el nombre del test asignado durante el entrenamiento.
        for file in search_files(dirname, 'csv'):
            # Se lee el dataframe
            data = pd.read_csv(file, sep=';')

            # Se recupera el nombre del modelo a partir del nombre asignado al log.
            data.loc[:, ['Model', 'Phase']] = get_filename(file).split('_')
            data.loc[:, ['Weights', 'FrozenLayers']] = file.split(os.sep)[-3:-1]

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

    @staticmethod
    def get_dataframe_from_preds(dirname: io) -> pd.DataFrame:
        """
        Función para crear un dataframe único a partir de las predicciones de cada red neuronal
        :param dirname: Directorio que almacena las predicciones de las redes
        :return: dataframe con los datos unidos
        """

        l = []
        # Se buscan todos los archivos csv del directorio
        for file in search_files(dirname, 'csv'):
            # Se lee el fichero
            df = pd.read_csv(file, sep=';')
            # Se crean las columnas weight y layer a partir del nombre del directorio
            df.loc[:, ['Weight', 'Layer']] = file.split(os.sep)[-3:-1]
            l.append(df.rename(columns={'PREDICTION': get_filename(file)}))
        # Se une toda la información
        return pd.concat(l, ignore_index=True).groupby(['PROCESSED_IMG', 'Weight', 'Layer'], as_index=False).first()

    def get_dataframe_from_dataset_excel(self) -> pd.DataFrame:
        """
        Función para leer el dataframe que contiene información del dataset que contiene los datos de train/val/test de
        experimento realizado
        """
        return pd.read_excel(self.conf.model_db_desc_csv, dtype=object, index_col=None)

    def get_threshold_opt(self, df: pd.DataFrame, models: list):
        """
        Función para obtener los thresholds optimos para cada red neuronal convolucional aplicanto una técnica de
        bootstraping.
        :param df: pandas dataframe con las predicciones de train y validación para cada red entrenada mediante
                   pesos de imagenet (columna Weight) y con las distintas estrategias de entrenamiento definidas (Layer)
        :param models: nombre de los modelos utilizados
        """

        # para cada inicialización de pesos y estrategia de entrenamiento se obtiene el threshold que maximiza
        # el AUC para los datos de validación
        thresholds = df[df.TRAIN_VAL == 'val'].groupby(['Weight', 'Layer'], as_index=False).apply(
            lambda x: pd.Series(
                {
                    c: apply_bootstrap(
                        x[['IMG_LABEL', c]].assign(LABEL=x.IMG_LABEL.map(self.labels)), metric=optimize_threshold,
                        true_col='LABEL', pred_col=c
                    )[0] for c in models
                }
            )
        )
        # Se registran en un csv los thresholds optimos
        thresholds.to_csv(get_path(self.conf.model_root_dir, 'thresholds.csv'), index=False, decimal=',', sep=';')

        # Se crea una columna con el label de cada modelo en función del umbral optimo obtenido
        for model in models:
            df.loc[:, f'{model}_LBL'] = np.nan

        # Se crea el label (maligno o benigno) a partir del humbral y la probabilidad de malignidad de cada muestra
        # teniendo en cuenta los pesos y la estrategia de entrenamiento
        for w, layer in product(df.Weight.unique(), df.Layer.unique()):

            # Se itera cada modelo para obtener los labels de cada instancia.
            for model in models:
                try:
                    thresh = thresholds.loc[(thresholds.Weight == w) & (thresholds.Layer == layer), model].values[0]
                except IndexError:
                    thresh = np.nan
                # Si el label es mayor al humbral, se asigna la clase malinga. En caso contraio, benigna.
                if not pd.isna(thresh):
                    df.loc[(df.Weight == w) & (df.Layer == layer), f'{model}_LBL'] = np.where(
                        df.loc[(df.Weight == w) & (df.Layer == layer), model] >= thresh, 'MALIGNANT', 'BENIGN')

    @staticmethod
    def plot_model_metrics(plot_params: List[dict], dirname: io = None, filename: io = None, plots_per_line: int = 2):
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
                       ylabel=plot_configuration['labels'].get('ylabel', plot_configuration['plot']['y']))

            # Configuración del título del gráfico
            if plot_configuration.get('title', False):
                ax.title.set(**plot_configuration['title'])

            # Configuración de la leyenda del gráfico
            if plot_configuration.get('legend', False):
                ax.legend(**plot_configuration['legend'])

            # Se elimina el grid del grafico
            ax.grid(False)
            sns.despine(ax=ax)

        # Ajuste de la figura y guardado de la imagen
        figure.tight_layout()
        figure.savefig(get_path(dirname, filename))

    def plot_confusion_matrix(self, df: pd.DataFrame, best_models: pd.DataFrame, models: list) -> None:
        """
            Función para crear las matrices de confusión de cada modelo
            :param df: dataframe con las predicciones de cada modelo
            :param best_models: dataframe con información de la estrategia de entrenamiento que mayor AUC obtiene
                                para los conjuntos de validación
            :param models: lista con los nombres de los modelos utilizados
        """
        # En función del número de modelos, se generan n hileras para graficar los resultados. Cada hilera contendrá
        # dos modelos.
        nrows = (len(models) // 2) + 1

        # Se itera para cada fase de entrenamiento/validación de cada modelo.
        for mode in df.TRAIN_VAL.unique():

            # Se crea la figura de matplotlib
            fig = plt.figure(figsize=(15, 10))

            # Se iteran los modelos
            for i, col in enumerate(models, 1):
                # Se obtiene para cada modelo el que mejor AUC ha obtenido para el set de datos de validación segun la
                # estrategia e inicialización de pesos
                layer = best_models.loc[best_models.CNN == col, 'FT'].values[0]
                weights = best_models.loc[best_models.CNN == col, 'WEIGHTS'].values[0]

                plt_data = df[(df.TRAIN_VAL == mode) & (df.Layer == layer) & (df.Weight == weights)]

                # Se crea un subplot.
                ax = fig.add_subplot(nrows, 2, i)

                # Se crea la tabla de contingencia a través de las clases verdaderas (columna true_label) y las
                # predecidas (columna definida por el nombre del modelo)..
                ct = pd.crosstab(plt_data.IMG_LABEL, plt_data[f'{col}_LBL'], normalize=False)

                # Se muestra la matriz de confusión.
                sns.set(font_scale=1)
                sns.heatmap(ct.reindex(sorted(ct.columns), axis=1), cmap="Blues", annot=True,
                            annot_kws={"size": 15}, fmt="d", cbar=False, ax=ax)

                # título y eje x del gráfico
                ax.set_title(f'Modelo: {col}\n{mode} ({weights} - {layer})', fontweight='bold', size=14)
                ax.set(xlabel='Predictions')

                # Se ajustan los subplots
                fig.tight_layout()
                # Se almacena la figura.
                fig.savefig(get_path(self.conf.model_viz_results_confusion_matrix_dir, f'{mode}.png'))

    def plot_metrics_table(self, df: pd.DataFrame, models: list, best: pd.DataFrame, class_metric: bool = True) -> None:
        """
            Función utilizada para generar una imagen con una tabla que contiene las métricas de AUC, accuracy,
            precision, recall y f1_score para entrenamiento y validación. Las métricas se calculan a partir del log de
            predicciones generado por un modelo. Se aplica un bootstrap para calcular las métricas

            :param df: pandas dataframe con las predicciones realizadas por cada modelo
            :param models: lista de modelos utilizados
            :param best: dataframe con la información de la mejor estrategia y la mejor inicialización para cada modelo
            :param class_metric: booleano que sirve para calcular las métricas de AUC, precision, recall, accuracy y f1
                                 de cada clase y modelo en caso de ser true. En caso contrario, se generarán las
                                 métricas de forma global para cada modelo.
        """

        # Lista con las métricas a representar
        metrics = ['AUC', 'accuracy', 'precision', 'recall', 'f1']

        # Se crea un dataset que contendrá las métricas AUC, accuracy, precision, recall y f1 para train y validación a
        # nivel general. En caso de tener el parametro class_metrics a True, las columnas del dataset serán un
        # multiindice con modelos y clases; en caso contrario, únicamente contendrá el nombre de los modelos.
        metric_df = pd.DataFrame(
            index=pd.MultiIndex.from_product([df.TRAIN_VAL.unique(), metrics]),
            columns=pd.MultiIndex.from_product([models, df.IMG_LABEL.unique().tolist()]) if class_metric
            else models
        )

        # Se asigna el nombre del índice
        metric_df.index.set_names(['mode', 'metric'], inplace=True)

        # Para cada modelo se recuperan losd atos de train y validación y test en caso de existir
        for phase, model in product(df.TRAIN_VAL.unique(), models):

            # Se obtienen las predicciones del mejor modelo
            layer = best.loc[best.CNN == model, 'FT'].values[0]
            weights = best.loc[best.CNN == model, 'WEIGHTS'].values[0]

            df_ = df[(df.TRAIN_VAL == phase) & (df.Layer == layer) & (df.Weight == weights)]

            if class_metric:
                # En caso de querer obtener las metricas de cada clase se itera sobre cada una de estas.
                for class_label in df_.IMG_LABEL.unique():
                    df_2 = df_[df_.TRAIN_VAL == phase][['IMG_LABEL', model]]

                    # Para poder generar las métricas deseadas, se considerará cada clase como verdadera
                    # asignandole el valor 1, y el resto de clases con el valor 0. De esta forma, se evaluará
                    # para cada clase. (Técnica one vs all)
                    map_dict = defaultdict(lambda: 0, {class_label: 1})

                    # Creación del dataset de métricas
                    metric_df.loc[(phase,), (model, class_label)] = [
                        '{:.2f} [{:.2f}, {:.2f}]'.format(
                            *apply_bootstrap(df_2.applymap(map_dict.get), true_col='IMG_LABEL', pred_col=model,
                                             metric=roc_auc_score)
                        ),
                        '{:.2f} [{:.2f}, {:.2f}]'.format(
                            *apply_bootstrap(df_2.applymap(map_dict.get), true_col='IMG_LABEL', pred_col=model,
                                             metric=accuracy_score)
                        ),
                        '{:.2f} [{:.2f}, {:.2f}]'.format(
                            *apply_bootstrap(df_2.applymap(map_dict.get), true_col='IMG_LABEL', pred_col=model,
                                             metric=precision_score, zero_division=0, average='weighted')
                        ),
                        '{:.2f} [{:.2f}, {:.2f}]'.format(
                            *apply_bootstrap(df_2.applymap(map_dict.get), true_col='IMG_LABEL', pred_col=model,
                                             metric=recall_score, zero_division=0, average='weighted')
                        ),
                        '{:.2f} [{:.2f}, {:.2f}]'.format(
                            *apply_bootstrap(df_2.applymap(map_dict.get), metric=f1_score, true_col='IMG_LABEL',
                                             pred_col=model, zero_division=0, average='weighted')
                        )
                    ]
            else:
                # Creación del dataset de métricas
                map_dict = {'BENIGN': 0, 'MALIGNANT': 1}

                df_2 = df_[df_.TRAIN_VAL == phase][['IMG_LABEL', f'{model}_LBL']]

                metric_df.loc[(phase,), model] = [
                    '{:.2f} [{:.2f}, {:.2f}]'.format(
                        *apply_bootstrap(df_2.applymap(map_dict.get), true_col='IMG_LABEL', pred_col=f'{model}_LBL',
                                         metric=roc_auc_score)
                    ),
                    '{:.2f} [{:.2f}, {:.2f}]'.format(
                        *apply_bootstrap(df_2.applymap(map_dict.get), true_col='IMG_LABEL', pred_col=f'{model}_LBL',
                                         metric=accuracy_score)
                    ),
                    '{:.2f} [{:.2f}, {:.2f}]'.format(
                        *apply_bootstrap(df_2.applymap(map_dict.get), true_col='IMG_LABEL', pred_col=f'{model}_LBL',
                                         metric=precision_score, zero_division=0, average='weighted')
                    ),
                    '{:.2f} [{:.2f}, {:.2f}]'.format(
                        *apply_bootstrap(df_2.applymap(map_dict.get), true_col='IMG_LABEL', pred_col=f'{model}_LBL',
                                         metric=recall_score, zero_division=0, average='weighted')
                    ),
                    '{:.2f} [{:.2f}, {:.2f}]'.format(
                        *apply_bootstrap(df_2.applymap(map_dict.get), metric=f1_score, true_col='IMG_LABEL',
                                         pred_col=f'{model}_LBL', zero_division=0, average='weighted')
                    )
                ]

        # se resetea el índice para poder mostrar en la tabla si las métricas son de entrenamiento o de
        # validación
        metric_df.reset_index(inplace=True)

        merge_rows = [
            [(i, 0) for i in range(l, l + len(metrics))] for l in
            range(metric_df.columns.nlevels, (len(metrics) + 1) * df_.TRAIN_VAL.nunique(), len(metrics))
        ]

        merge_cols = [
            [(0, i) for i in range(l, l + df_.IMG_LABEL.nunique())]
            for l in range(2, (len(models) + 1) * df_.IMG_LABEL.nunique(), df_.IMG_LABEL.nunique())
        ]
        merge_cells = [[(0, 0), (1, 0)], [(0, 1), (1, 1)], *merge_rows, *merge_cols] if class_metric else None

        fig, _ = render_mpl_table(metric_df, merge_pos=merge_cells, header_rows=2)
        filename = f'{ENSEMBLER_CONFIG}_model_metrics{"_marginal" if class_metric else ""}.jpg'
        fig.savefig(get_path(self.conf.model_viz_results_metrics_dir, filename))

    def plot_accuracy_plots(self, df: pd.DataFrame, models: list, hue: str, title: str, img_name: str):
        """
            Función para graficar en función de cada estrategia e inicialización de pesos las metricas de AUC,
            precision, accuracy, recall y f1 aplicando una técnica de bootstrap. La representaciíon será mediante
            gráficos de barras
            :param df: pandas dataframe con las predicciones y los valores reales de cada instancia
            :param models: lista con los modelos a representar
            :param hue: nombre de la clase para representar las métricas
            :param title: titulo del grafico
            :param img_name: nombre de la imagen a almacenar
        """
        # Diccionario con las métricas a representar
        metrics = dict(
            AUC=lambda d, y: apply_bootstrap(d, true_col='IMG_LABEL', pred_col=y, metric=roc_auc_score),
            ACCURACY=lambda d, y: apply_bootstrap(d, true_col='IMG_LABEL', pred_col=y, metric=accuracy_score),
            PRECISION=lambda d, y: apply_bootstrap(
                d, true_col='IMG_LABEL', pred_col=y, metric=precision_score, zero_division=0, average='weighted'
            ),
            RECALL=lambda d, y: apply_bootstrap(
                d, true_col='IMG_LABEL', pred_col=y, metric=recall_score, zero_division=0, average='weighted'
            ),
            F1=lambda d, y: apply_bootstrap(
                d, true_col='IMG_LABEL', pred_col=y, metric=f1_score, zero_division=0, average='weighted')
        )

        # Se itera cada métrica para obtener los gráficos
        for metric, metric_clb in metrics.items():

            fig = plt.figure(figsize=(15, 10))

            # Se iteran las fases de entrenamiento/validación/test para obtener cada métrica en función de la
            # estrategia y la iniciaizción de los pesos
            for i, phase in enumerate(df.TRAIN_VAL.unique(), 1):

                # Se obtiene cada métric agrupando por la estrategia de entrenamiento y la inicialización de los pesos
                df_grouped = df[df.TRAIN_VAL == phase].groupby(['Weight', 'Layer'], as_index=False). \
                    apply(
                    lambda f: pd.Series({
                        md: "{:.4f}-{:.4f}-{:.4f}".format(
                            *metric_clb(f[['IMG_LABEL', f'{m}_LBL']].applymap(self.labels.get), f'{m}_LBL')
                        ) for md in models
                    })
                )

                # Se obtienen las métricas de cada modelo
                for m in models:
                    df_grouped.loc[:, [m, f'{m}_l', f'{m}_u']] = (
                            df_grouped[m].str.split('-', expand=True).applymap(float) * 100
                    ).values
                # Se crea un único dataframe
                df_melt = pd.concat(
                    objs=[
                        pd.melt(df_grouped, id_vars=['Weight', 'Layer'], value_vars=models, var_name='model',
                                value_name=metric),
                        pd.melt(df_grouped, id_vars=['Weight', 'Layer'], value_vars=[f'{m}_u' for m in models],
                                var_name='model', value_name=f'{metric}_u')[[f'{metric}_u']],
                        pd.melt(df_grouped, id_vars=['Weight', 'Layer'], value_vars=[f'{m}_l' for m in models],
                                var_name='model', value_name=f'{metric}_l')[[f'{metric}_l']]
                    ], axis=1
                )

                # Creación del gráfico
                axes = fig.add_subplot(1, 2, i)

                sns.barplot(x='model', y=metric, hue=hue, data=df_melt, ax=axes)
                # Se crean las barras de error de cada gráfico obtenidas por la técnica de bootrstrap
                for x, (lgnd, patch) in zip(axes.patches, product(axes.get_legend_handles_labels()[1],
                                                                  axes.xaxis.get_ticklabels())):
                    cond = (df_melt.model == patch.get_text()) & (df_melt[hue] == lgnd)

                    low, high = df_melt.loc[cond, [f'{metric}_l', f'{metric}_u']].values.tolist()[0]
                    plt.vlines(x.get_x() + x.get_width() * 0.5, ymin=low, ymax=high, edgecolor='black')

                axes.set(xlabel=phase, ylim=[0, 100])

                fig.tight_layout()
                fig.suptitle(title, y=1.05, fontsize=14, fontweight='bold')
                fig.savefig(get_path(self.conf.model_viz_results_dir, metric, ENSEMBLER_CONFIG, f'{img_name}.png'))

    def plot_auc_roc_curves(self, df: pd.DataFrame, best_models: pd.DataFrame, models: list):
        """
        Función utilizada para representar las curvas ROC de los mejores clasificadores individuales
        :param df: pandas dataframe con la información de las predicciones para cada estrategia y inicialización de
                   pesos
        :param best_models: pandas dataframe con la inforamción de las mejores estrategias e inicialización de pesos
        :param models: lista con los nombres de los modelos a representar
        """

        # Se itera para cada fase de entrenamiento/validación de cada modelo.
        for model in models:

            # Se crea la figura de matplotlib
            fig = plt.figure(figsize=(15, 10))

            # Se representa un clasificador sin skills
            plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

            # Se filtra la mejor estrategia e inicializción de pesos (mayor AUC) para cada modelo
            layer = best_models.loc[best_models.CNN == model, 'FT'].values[0]
            weights = best_models.loc[best_models.CNN == model, 'WEIGHTS'].values[0]

            # Se representan las curvas auc de entrenamiento y validación
            for phase in df.TRAIN_VAL.unique():

                plt_data = df[(df.TRAIN_VAL == phase) & (df.Layer == layer) & (df.Weight == weights)]

                fpr, tpr, thresholds = roc_curve(plt_data['IMG_LABEL'].map(self.labels), plt_data[model])

                if phase == 'train':
                    ix = np.argmax(tpr - fpr)
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                plt.plot(fpr, tpr, marker='.', label=phase)

            # Titulos de los ejes y leyenda
            plt.xlabel('False Positive Rate')
            plt.ylabel('True positve rate')
            plt.legend()

            # Se ajustan los subplots
            fig.tight_layout()
            # Se almacena la figura.
            fig.savefig(get_path(self.conf.model_viz_results_auc_curves_dir, f'{model}.png'))

    def get_model_time_executions(self, summary_dir: io):
        """
        Función para representar los tiempos de ejecución por epoca para cada estrategia y red
        :param summary_dir: directorio que contiene el archivo de summary generado durante el entrenamiento de las redes
        """

        # Lectura del summary dir
        data = pd.read_csv(search_files(summary_dir, 'csv', in_subdirs=False)[0], sep=';', decimal='.')

        fig, ax = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

        # Para cada estrategia de fine tunning se obtienen los tiempos de entrenamiento normalizados de cada modelo
        # individual
        for weights, axes in zip(data.weights.unique(), ax.flatten()):
            data_filtered = data[data.weights == weights]

            # Calculo de tiempos de entrenamiento por epoca
            data_plot = data_filtered.groupby(['cnn', 'FT'], as_index=False). \
                apply(lambda x: pd.Series(x['time'].sum() / x['epochs'].sum(), index=['time_eps']))

            # Representación del gráfico de barras
            sns.barplot(x='cnn', y='time_eps', hue='FT', data=data_plot, ax=axes)

            # Ejes de los gráficos
            axes.set_title(f'Tiempo de entrenamiento con inicialización de pesos: {weights}')
            axes.set(ylabel='Tiempo / epocas (seg)', xlabel='Capas entrenables')

        fig.tight_layout()
        fig.savefig(get_path(self.conf.model_viz_results_dir, 'Comparación tiempos entrenamiento.jpg'))

    def get_model_logs_metrics(self, logs_dir: io):
        """
        Función para generar las métricas de accuracy, precision, recall, auc, f1 score y loss a partir de los logs
        generados durante la fase de entrenamiento de un modelo
        :param logs_dir: directorio en el cual se almacenan los logs
        """

        # Se recupera un dataframe a partir del directorio de logs que contendrá las métricas
        data = self.get_dataframe_from_logs(logs_dir, self.metrics)

        # Se itera sobre las fases con las que se ha entrenado cada modelo
        for weights, layers, phase in product(data.Weights.unique().tolist(), data.FrozenLayers.unique().tolist(),
                                              data.Phase.unique().tolist()):
            # Filtrado de datos
            data_filtered = data[(data.Weights == weights) & (data.Phase == phase) & (data.FrozenLayers == layers)]

            if len(data_filtered) > 0:
                # Se crea la gráfica correspondiente
                self.plot_model_metrics(
                    dirname=self.conf.model_viz_results_model_history_dir,
                    filename=f'Model_history_train_{phase}_{weights}_{layers}.jpg',
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
                                'text': f'{title}',
                                'fontsize': 12,
                                'fontweight': 'bold'
                            },
                            'legend': {
                                'fontsize': 'x-small',
                                'frameon': False,
                                'framealpha': 1
                            }
                        } for title, metric, ylabel, in zip(
                            ['AUC Modelos', 'Precisión Modelos', 'Recall Modelos', 'F1_Score Modelos',
                             'Exactitud Modelos', 'Pérdidas Modelos'], self.metrics,
                            ['AUC', 'Precisión', 'Recall', 'F1', 'Exactitud', 'Pérdidas'])
                    ]
                )

    def get_model_predictions_metrics(self, cnn_predictions_dir: io, ensembler_predictions_dir: io):
        """
        Función que genera el pipeline para obtener los thresholds, graficar las matrices de confusión, las tablas de
        métricas y los gráficos de barras de las métricas de cada modelo generado.

       :param cnn_predictions_dir: directorio en el que se almacenan las predicciones de las redes
       :param ensembler_predictions_dir: directorio en el que se almacenan las predicciones del model enembling
       """

        # Lectura de los datos y unificación en un dataframe
        df = pd.merge(
            left=self.get_dataframe_from_preds(dirname=cnn_predictions_dir),
            right=self.get_dataframe_from_preds(dirname=ensembler_predictions_dir).assign(Weight='imagenet',
                                                                                          Layer='ALL'),
            on=['PROCESSED_IMG', 'IMG_LABEL', 'TRAIN_VAL', 'Weight', 'Layer'],
            how='left'
        )

        # Se obtienen las mejores estrategias de ajuste fino e inicialización de pesos para cada modelo generadas
        # al entrenar el model ensembling
        best_model = pd.concat(
            objs=[
                pd.read_csv(
                    get_path(self.conf.model_store_ensembler_dir, ENSEMBLER_CONFIG, 'Selected CNN Report.csv'), ';'
                ),
                pd.DataFrame(
                    [['RandomForest', 'ALL', 'imagenet', '-', 0]], columns=['CNN', 'FT', 'WEIGHTS', 'TRAIN_VAL', 'AUC']
                )
            ], axis=0
        )

        # Se recupera la lista de modelos
        models = [c for c in df.columns if c not in ['PROCESSED_IMG', 'IMG_LABEL', 'TRAIN_VAL', 'Weight', 'Layer']]

        # Se obtienen los thresholds de decisión optimos para cada modelo
        self.get_threshold_opt(df, models)

        # Se obtienen las curvas roc de los modelos
        self.plot_auc_roc_curves(df=df, models=models, best_models=best_model)

        # Se representan las matrices de confusión de los modelos
        self.plot_confusion_matrix(df=df, models=models, best_models=best_model)

        # Se representan las tablas de métricas de los modelos
        self.plot_metrics_table(df, models, best_model, class_metric=False)

        # Plots de barras de las métricas para las estrategias inicializadas en imagenet
        self.plot_accuracy_plots(
            df[df.Weight == 'imagenet'], models[:-1], hue='Layer', img_name='Frozen Layers Accuracy',
            title='Impact of the fraction of convolutional blocks fine-tuned on CNN performance'
        )

        # Comparación de la performance de los modelos en función de la inicialización de pesos random o imagenet
        self.plot_accuracy_plots(
            df[df.Layer == 'ALL'], models[:-1], hue='Weight', img_name='Weight Init Accuracy',
            title='Random Initialization vs Imagenet'
        )

    def get_data_augmentation_examples(self) -> None:
        """
        Función que permite generar un ejemplo de cada tipo de data augmentation aplicado
        """

        # Se recuperan las instancias del excel de entrenamiento
        df = self.get_dataframe_from_dataset_excel()

        # Se obtiene una instancia aleatoria
        example_imag = df.loc[random.sample(df.index.tolist(), 1)[0], 'PROCESSED_IMG']

        # Se lee la imagen del path de ejemplo
        image = load_img(example_imag)
        # Se transforma la imagen a formato array
        image = img_to_array(image)

        # Se añade una dimensión para obtener el dato de forma (1, width, height, channels)
        image_ori = np.expand_dims(image, axis=0)

        # Figura y subplots de matplotlib. Se crea una figura con un grid de 4 columnas y el número de hileras dependera
        # de las transforamciones de data aumentation aplicadas
        elements = len(CLASSIFICATION_DATA_AUGMENTATION_FUNCS.keys()) + 1
        cols = 4
        rows = elements // cols + (elements % cols > 0)
        fig = plt.figure(figsize=(15, 4 * rows))

        # Se representa la imagen original en el primer subplot.
        plot_image(img=image_ori, title='Imagen Original', ax_=fig.add_subplot(rows, cols, 1))

        # Se iteran las transformaciones
        for i, (transformation_name, transformation) in enumerate(CLASSIFICATION_DATA_AUGMENTATION_FUNCS.items(), 2):
            # Se crea al datagenerator con exclusivamente la transformación a aplicar.
            transformation.p = 1
            img = np.expand_dims(Compose([transformation])(image=image)['image'], axis=0)

            # Se recupera la imagen transformada mediante next() del método flow del objeto datagen
            plot_image(img=img, title=transformation_name, ax_=fig.add_subplot(rows, cols, i))

        # Se ajusta la figura
        fig.tight_layout()

        # Se almacena la figura
        plt.savefig(get_path(self.conf.model_viz_data_augm_dir, f'{get_filename(example_imag)}.png'))

    def get_eda_from_df(self) -> None:
        """
        Función para generar el pipeline del data exploratori de los datos para entrenar los modelos.
        """

        df = self.get_dataframe_from_dataset_excel()

        print(f'{"-" * 75}\n\tGenerando análisis del dataset\n{"-" * 75}')
        title = 'Distribución clases según orígen'
        file = get_path(self.conf.model_viz_eda_dir, f'{title}.png')
        create_countplot(x='DATASET', hue='IMG_LABEL', data=df, title=title, file=file)

        title = 'Distribución clases'
        file = get_path(self.conf.model_viz_eda_dir, f'{title}.png')
        create_countplot(x='IMG_LABEL', data=df, title=title, file=file)

        title = 'Distribución clases segun train-val'
        file = get_path(self.conf.model_viz_eda_dir, f'{title}.png')
        create_countplot(x='TRAIN_VAL', hue='IMG_LABEL', data=df, title=title, file=file, norm=True)

        title = 'Distribución clases segun patología'
        file = get_path(self.conf.model_viz_eda_dir, f'{title}.png')
        create_countplot(x='ABNORMALITY_TYPE', hue='IMG_LABEL', data=df, title=title, file=file, norm=True)
        print(f'{"-" * 75}\n\tAnálisis del dataset finalizado en {self.conf.model_viz_eda_dir}\n{"-" * 75}')

    def get_preprocessing_examples(self) -> None:
        """
        Función para plotar cada uno de los pasos de transformación y preprocesado aplicados.
        """

        # Lectura de los dtaos
        df = self.get_dataframe_from_dataset_excel().assign(example_dir=None)
        photos = []

        # Se obtienen 5 muestras aleatorias de cada dataset
        for dataset in df.DATASET.unique():
            photos += random.sample(df[df.DATASET == dataset].index.tolist(), 5)

        # Se crean los directorios de salida
        df.loc[photos, 'example_dir'] = df.loc[photos, :].apply(
            lambda x: get_path(self.conf.model_viz_preprocesing_dir, x.DATASET, get_filename(x.PROCESSED_IMG),
                               f'{get_filename(x.PROCESSED_IMG)}.png'),
            axis=1
        )

        # Se llama a la función de preprocesado dependiendo de si se trata de una clasificación a nivel de imágenes o de
        # parches
        if self.img_type == 'COMPLETE_IMAGE':
            for _, r in df[df.example_dir.notnull()].iterrows():
                full_image_pipeline([r.CONVERTED_IMG, r.example_dir, True])
        elif self.img_type == 'PATCHES':
            for _, r in df[df.example_dir.notnull()].iterrows():
                crop_image_pipeline([r.CONVERTED_IMG, r.example_dir, r.X_MAX, r.Y_MAX, r.X_MIN, r.Y_MIN, True])
        else:
            raise ValueError(f"Function {self.img_type} doesn't defined")
