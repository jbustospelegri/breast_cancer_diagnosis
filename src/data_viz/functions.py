from itertools import product
from keras_preprocessing.image import array_to_img
from pandas import DataFrame
from typing import io

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def create_countplot(data: DataFrame, file: io, x: str, hue: str = None, title: str = '', annotate: bool = True,
                     norm: bool = False):

    # Figura de matplotlib para almacenar el gráfico
    plt.figure(figsize=(15, 5))

    # Gráfico de frecuencias
    ax = sns.countplot(x=x, data=data, hue=hue, palette=sns.light_palette((210, 90, 60), input='husl', reverse=True))

    # Se realizan las anotaciones de los valores de frecuencia y frecuencia normalizda para cada valor de la
    # variable objetivo.
    if annotate:
        ax.patches.sort(key=lambda x: x.get_x())
        for p, (l, _) in zip(ax.patches, product(ax.xaxis.get_ticklabels(),
                                                 [*ax.get_legend_handles_labels()[1],
                                                  *ax.xaxis.get_ticklabels()][:data[x].nunique()])):
            txt = '{a:.0f} ({b:.2f} %)'.format(
                a=p.get_height(),
                b=(p.get_height() / (len(data[data[x] == l.get_text()]) if norm else len(data))) * 100
            )
            ax.annotate(txt, xy=(p.get_x() + p.get_width() * 0.5, p.get_height()), va='center', ha='center',
                        clip_on=True, xycoords='data', xytext=(0, 7), textcoords='offset points')

    # Título del gráfico
    if title:
        ax.set_title(title, fontweight='bold', size=14)

    # Se elimina el label del eje y.
    ax.set(ylabel='')

    # Se almacena la figura
    plt.savefig(file)


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14, header_color='#40466e', row_colors=None,
                     edge_color='w', bbox=None, header_columns=0, ax=None, **kwargs):
    """
    Función utilizada para renderizar un dataframe en una tabla de matplotlib. Función recuperada de
    https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure
    """
    if bbox is None:
        bbox = [0, 0, 1, 1]

    if row_colors is None:
        row_colors = ['#f1f1f2', 'w']

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax.get_figure(), ax


def plot_image(img: np.ndarray, title: str, ax_: plt.axes):
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
