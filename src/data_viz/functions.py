from itertools import product
from tensorflow.keras.preprocessing.image import array_to_img
from pandas import DataFrame
from typing import io, List

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
        ax_ = list(ax.patches)
        ax_.sort(key=lambda x: x.get_x())
        for p, (l, _) in zip(ax_, product(ax.xaxis.get_ticklabels(), [*ax.get_legend_handles_labels()[1],
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
    ax.yaxis.grid(True)

    sns.despine(ax=ax, left=True)

    # Se almacena la figura
    plt.savefig(file)


def merge_cells(table: plt.table, cells: List[tuple]):
    '''
    Merge N matplotlib.Table cells

    Parameters
    -----------
    table: matplotlib.Table
        the table
    cells: list[set]
        list of sets od the table coordinates
        - example: [(0,1), (0,0), (0,2)]

    Notes
    ------
    https://stackoverflow.com/a/53819765/12684122
    '''
    cells_array = [np.asarray(c) for c in cells]
    h = np.array([cells_array[i + 1][0] - cells_array[i][0] for i in range(len(cells_array) - 1)])
    v = np.array([cells_array[i + 1][1] - cells_array[i][1] for i in range(len(cells_array) - 1)])

    # if it's a horizontal merge, all values for `h` are 0
    if not np.any(h):
        # sort by horizontal coord
        cells = np.array(sorted(list(cells), key=lambda v: v[1]))
        edges = ['BTL'] + ['BT'] * (len(cells) - 2) + ['BTR']
    elif not np.any(v):
        cells = np.array(sorted(list(cells), key=lambda h: h[0]))
        edges = ['TRL'] + ['RL'] * (len(cells) - 2) + ['BRL']
    else:
        raise ValueError("Only horizontal and vertical merges allowed")

    for cell, e in zip(cells, edges):
        table[cell[0], cell[1]].visible_edges = e

    for cell in cells[1:]:
        table[cell[0], cell[1]].get_text().set_visible(False)


def render_mpl_table(data, font_size=14, merge_pos: List[List[tuple]] = None, header_rows: int = 1):
    """
    Función utilizada para renderizar un dataframe en una tabla de matplotlib. Función recuperada de
    https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure
    """

    col_width = 3
    row_height = 0.625
    row_colors = ['#f1f1f2', 'w']

    size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
    fig, ax = plt.subplots(figsize=size)
    ax.axis('off')
    header_columns = data.columns.nlevels

    table = ax.table(
        cellText=np.vstack([*[data.columns.get_level_values(i) for i in range(0, header_columns)], data.values]),
        bbox=[0, 0, 1, 1], cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for k, cell in table._cells.items():
        if k[0] < header_columns or k[1] < header_rows:
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    if merge_pos is not None:
        for cells in merge_pos:
            merge_cells(table, cells)

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
