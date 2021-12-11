import os
import sys
from typing import io

from src.algorithms.metrics import f1_score
from src.utils.functions import get_path

import cv2

"""
    CONFIGURACION DEL EXPERIMENTO
"""
# Los valores disponibles son PATCHES, COMPLETE_IMAGE, MASK
EXPERIMENT = 'PATCHES'

"""
    CONFIGURACION DEL DATASET
"""
TRAIN_DATA_PROP: float = 0.8

"""
    CONFIGURACION DATA AUGMENTATION
"""
DATA_AUGMENTATION_FUNCS: dict = {
    'horizontal_flip': True,
    'vertical_flip': True,
    # 'shear_range': 0.1,
    'rotation_range': 270,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'brightness_range': (0.6, 1),
    # 'zoom_range': [0.8, 1.2],
}

"""
    CONFIGURACION DE EJECUCIONES DE LOS MODELOS
"""
EPOCHS: int = 100
WARM_UP_EPOCHS: int = 30
WARM_UP_LEARNING_RATE: float = 1e-3
LEARNING_RATE: float = 1e-4

BATCH_SIZE: int = 18
SEED: int = 81

METRICS = {
    'AUC': 'AUC',
    'Precision': 'Precision',
    'Recall': 'Recall',
    'F1 Score': f1_score
}

"""
    CONFIGURACION PARA EL GRADIENT BOOSTING
"""
N_ESTIMATORS = 20
MAX_DEPTH = 3
XGB_CONFIG = 'CONF1'
XGB_COLS = {
    'CONF1': [],
    'CONF2': ['BREAST', 'BREAST_VIEW', 'BREAST_DENSITY']
}

"""
    CONFIGURACIÓN DE PREPROCESADO DE IMAGENES
"""
IMG_SHAPE: tuple = (1024, 1024)
PATCH_SIZE: int = 300

CROP_CONFIG: str = 'CONF0'
CROP_PARAMS: dict = {
    'CONF0': {
        'N_BACKGROUND': 5,
        'N_ROI': 5,
        'OVERLAP': 1,
        'MARGIN': 1.2
    },
    'CONF1': {
        'N_BACKGROUND': 0,
        'N_ROI': 1,
        'OVERLAP': 1,
        'MARGIN': 1
    },
    'CONF3': {
        'N_BACKGROUND': 1,
        'N_ROI': 1,
        'OVERLAP': 1,
        'MARGIN': 1.4
    },
    'CONF4': {
        'N_BACKGROUND': 0,
        'N_ROI': 1,
        'OVERLAP': 1,
        'MARGIN': 1.4
    },
}

PREPROCESSING_CONFIG: str = 'CONF2'
PREPROCESSING_FUNCS: dict = {
    'CONF1': {
        'CROPPING_1': {
            'left': 0.01,
            'right': 0.01,
            'top': 0.04,
            'bottom': 0.04
        },
        'REMOVE_NOISE': {
            'ksize': 3
        },
        'REMOVE_ARTIFACTS': {
            'bin_kwargs': {
                'thresh': 'otsu',
                'threshval': 30
            },
            'mask_kwargs': {
                'kernel_shape': cv2.MORPH_ELLIPSE,
                'kernel_size': (20, 10),
                'operations': [(cv2.MORPH_OPEN, None), (cv2.MORPH_DILATE, 2)]
            },
            'contour_kwargs': {
                'convex_contour': True,
            },
            'crop_box': True,
        },
        'NORMALIZE_BREAST': {
            'type_norm': 'truncation'
        },
        'FLIP_IMG': {
            'orient': 'left'
        },
        'ECUALIZATION': {
            'clahe_1': {'clip': 2},
            'clahe_2': {'clip': 3},
        },
        'SQUARE_PAD': True,
        'RESIZING': {
            'size': (1024, 1024)
        },
        'CROPPING_2': {
            'left': 0.05,
            'right': 0,
            'top': 0,
            'bottom': 0
        },
    },
    'CONF2': {
        'CROPPING_1': {
            'left': 0.01,
            'right': 0.01,
            'top': 0.04,
            'bottom': 0.04
        },
        'REMOVE_NOISE': {
            'ksize': 3
        },
        'REMOVE_ARTIFACTS': {
            'bin_kwargs': {
                'thresh': 'constant',
                'threshval': 30
            },
            'mask_kwargs': {
                'kernel_shape': cv2.MORPH_ELLIPSE,
                'kernel_size': (20, 10),
                'operations': [(cv2.MORPH_OPEN, None), (cv2.MORPH_DILATE, 2)]
            },
            'contour_kwargs': {
                'convex_contour': False,
            }
        },
        'NORMALIZE_BREAST': {
            'type_norm': 'min_max'
        },
        'FLIP_IMG': {
            'orient': 'left'
        },
        'ECUALIZATION': {
            'clahe_1': {'clip': 2},
        },
        'SQUARE_PAD': True,
        'RESIZING': {
            'size': IMG_SHAPE
        },
        'CROPPING_2': {
            'left': 0.05,
            'right': 0,
            'top': 0,
            'bottom': 0
        },
    },
}


"""
    CARPETAS PRINCIPALES DEL PROGRAMA
"""
WORKING_DIRECTORY = sys._MEIPASS if getattr(sys, 'frozen', False) else os.getcwd()
RAW_DATA_PATH: io = get_path(WORKING_DIRECTORY, 'data', '00_RAW') if getattr(sys, 'frozen', False) else \
    get_path('..', 'data', '00_RAW')
CONVERTED_DATA_PATH: io = get_path(WORKING_DIRECTORY, 'data', '01_CONVERTED') if getattr(sys, 'frozen', False) \
    else get_path('..', 'data', '01_CONVERTED')
PROCESSED_DATA_PATH: io = get_path(WORKING_DIRECTORY, 'data', '02_PROCESSED') if getattr(sys, 'frozen', False) \
    else get_path('..', 'data', '02_PROCESED')
OUTPUT_DATA_PATH: io = get_path(os.getcwd(), 'OUTPUTS') if getattr(sys, 'frozen', False) \
    else get_path('..', 'data', '03_OUTPUT')
MODEL_DATA_PATH: io = get_path(WORKING_DIRECTORY, 'models') if getattr(sys, 'frozen', False) \
    else get_path('..', 'models')
LOGGING_DATA_PATH: io = get_path(os.getcwd(), 'LOGS') if getattr(sys, 'frozen', False) else get_path('..', 'logging')

"""
    CARPETAS CON LOS DATASETS
"""
CBIS_DDSM_PATH: io = get_path(RAW_DATA_PATH, 'CBIS-DDSM')
MIAS_PATH: io = get_path(RAW_DATA_PATH, 'MIAS')
INBREAST_PATH: io = get_path(RAW_DATA_PATH, 'INBreast')

"""
    FICHEROS CON LA METADATA DE LAS IMAGENES
"""
CBIS_DDSM_CALC_CASE_DESC_TEST: io = get_path(CBIS_DDSM_PATH, 'calc_case_description_test_set.csv')
CBIS_DDSM_CALC_CASE_DESC_TRAIN: io = get_path(CBIS_DDSM_PATH, 'calc_case_description_train_set.csv')
CBIS_DDSM_MASS_CASE_DESC_TEST: io = get_path(CBIS_DDSM_PATH, 'mass_case_description_test_set.csv')
CBIS_DDSM_MASS_CASE_DESC_TRAIN: io = get_path(CBIS_DDSM_PATH, 'mass_case_description_train_set.csv')
INBREAST_CASE_DESC: io = get_path(INBREAST_PATH, 'INbreast.xls')
MIAS_CASE_DESC: io = get_path(MIAS_PATH, 'Info.txt')

"""
    CARPETAS QUE CONTIENEN LAS IMAGENES
"""
CBIS_DDSM_DB_PATH: io = get_path(CBIS_DDSM_PATH, 'ALL')
MIAS_DB_PATH: io = get_path(MIAS_PATH, 'ALL')
INBREAST_DB_PATH: io = get_path(INBREAST_PATH, 'ALL')

"""
    CARPETA CON INFORMACIÓN DE LOS ROI's
"""
INBREAST_DB_XML_ROI_PATH = get_path(INBREAST_PATH, 'AllXML')

"""
    CARPETAS CON IMAGENES CONVERTIDAS
"""
CBIS_DDSM_CONVERTED_DATA_PATH: io = get_path(CONVERTED_DATA_PATH, 'CBIS_DDSM')
MIAS_CONVERTED_DATA_PATH: io = get_path(CONVERTED_DATA_PATH, 'MIAS')
INBREAST_CONVERTED_DATA_PATH: io = get_path(CONVERTED_DATA_PATH, 'INBreast')

"""
    CARPETAS CON LAS IMAGENES PROCESADAS
"""
CBIS_DDSM_PREPROCESSED_DATA_PATH: io = get_path(PROCESSED_DATA_PATH, PREPROCESSING_CONFIG, 'CBIS_DDSM')
MIAS_PREPROCESSED_DATA_PATH: io = get_path(PROCESSED_DATA_PATH, PREPROCESSING_CONFIG, 'MIAS')
INBREAST_PREPROCESSED_DATA_PATH: io = get_path(PROCESSED_DATA_PATH, PREPROCESSING_CONFIG, 'INBreast')

"""
    CARPETAS DE LA INTERFAZ GRÁFICA
"""
GUI_CSS_PATH = get_path(WORKING_DIRECTORY, 'static', 'css')


"""
    CARPETAS DE RESULTADOS DEL MODELO 
"""


class ModelConstants:

    def set_model_name(self, name: str) -> None:
        self.model_root_dir: io = get_path(MODEL_DATA_PATH, name)

        self.model_db_desc_csv: io = get_path(self.model_root_dir, 'dataset.xlsx')
        self.model_db_processing_info_file: io = get_path(self.model_root_dir, 'preprocess_config.json')

        self.model_store_dir: io = get_path(self.model_root_dir, 'STORED_MODELS')
        self.model_store_cnn_dir: io = get_path(self.model_store_dir, 'CNN')
        self.model_store_xgb_dir: io = get_path(self.model_store_dir, 'GRADIENT_BOOSTING')

        self.model_log_dir: io = get_path(self.model_root_dir, 'TRAIN_LOGS')
        self.model_summary_dir: io = get_path(self.model_root_dir, 'SUMMARY_MODELS')
        self.model_summary_train_csv: io = get_path(self.model_root_dir, 'SUMMARY_MODELS', 'train_summary.csv')

        self.model_predictions_dir: io = get_path(self.model_root_dir, 'PREDICTIONS')
        self.model_predictions_cnn_dir = get_path(self.model_predictions_dir, 'CNN')
        self.model_predictions_xgb_dir = get_path(self.model_predictions_dir, 'GRADIENT_BOOSTING')

        self.model_data_viz_dir: io = get_path(self.model_root_dir, 'DATA_VIZ')

        self.model_viz_preprocesing_dir: io = get_path(self.model_root_dir, 'PREPROCESSING_VIZ', PREPROCESSING_CONFIG)
        self.model_viz_eda_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'DATASET_EDA')
        self.model_viz_train_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'TRAIN_PHASE')
        self.model_viz_data_augm_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'DATA_AUGMENTATION_EXAMPLES')

        self.model_viz_results_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'RESULTS')
        self.model_viz_results_model_history_dir: io = get_path(self.model_viz_results_dir, 'MODEL_HISTORY')
        self.model_viz_results_confusion_matrix_dir: io = get_path(self.model_viz_results_dir, 'CONFUSION_MATRIX')
        self.model_viz_results_accuracy_dir: io = get_path(self.model_viz_results_dir, 'ACCURACY')
        self.model_viz_results_metrics_dir: io = get_path(self.model_viz_results_dir, 'METRICS')


MODEL_FILES = ModelConstants()