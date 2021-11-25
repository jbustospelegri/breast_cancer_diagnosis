from typing import Mapping, io, Union

from algorithms.metrics import f1_score
from utils.functions import get_path

import cv2


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
    'shear_range': 0.1,
    'zoom_range': 0.2,
}

"""
    CONFIGURACION DE EJECUCIONES DE LOS MODELOS
"""

EPOCHS: int = 100
WARM_UP_EPOCHS: int = 30
LEARNING_RATE: float = 1e-3

BATCH_SIZE: int = 16
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
    'CONF2': ['BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'ABNORMALITY_TYPE']
}

"""
    CONFIGURACIÓN DE PREPROCESADO DE IMAGENES
"""
IMG_SHAPE: int = 512
PREPROCESSING_CONFIG: str = 'CONF1'
PREPROCESSING_FUNCS: Mapping[str, Mapping[str, Union[bool, Mapping[str, Union[str, int, float, tuple, dict]]]]] = {
    'CONF1': {
        'CROPPING_1': {
            'left': 0.01,
            'right': 0.01,
            'top': 0.04,
            'bottom': 0.04
        },
        'MIN_MAX_NORM': {
            'min': 0,
            'max': 255
        },
        'REMOVE_ARTIFACTS_BLACK_BORDERS': {
            'bin_kwargs': {
                'thresh': 'constant',
                'threshval': 1
            },
            'mask_kwargs': {
                'kernel_shape': cv2.MORPH_RECT,
                'kernel_size': (20, 20),
                'operations': [(cv2.MORPH_CLOSE, None)]
            }
        },
        'REMOVE_ARTIFACTS': {
            'bin_kwargs': {
                'thresh': 'adaptative',
                'size': 7,
                'method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                'c': 2
            },
            'mask_kwargs': {
                'kernel_shape': cv2.MORPH_ELLIPSE,
                'kernel_size': (20, 20),
                'operations': [(cv2.MORPH_CLOSE, None), (cv2.MORPH_OPEN, 2)]
            }
        },
        'FLIP_IMG': {
            'orient': 'left'
        },
        # 'REMOVE_NOISE': {
        #     # 'additive_noise': {
        #     #     'ksize': 3
        #     # },
        #     # 'multiplicative_noise': {
        #     #     'gh': 2,
        #     #     'gl': 0.5,
        #     #     'd0': 30,
        #     #     'c': 1
        #     # },
        # },
        'ECUALIZATION': {
            'clahe_1': {'clip': 2}
        },
        'SQUARE_PAD': True,
        'RESIZING': {
            'size': (IMG_SHAPE, IMG_SHAPE)
        }
    },
    'CONF2': {
        'CROPPING_1': {
            'left': 0.01,
            'right': 0.01,
            'top': 0.04,
            'bottom': 0.04
        },
        'MIN_MAX_NORM': {
            'min': 0,
            'max': 255
        },
        'REMOVE_ARTIFACTS_BLACK_BORDERS': {
            'bin_kwargs': {
                'thresh': 'constant',
                'threshval': 1
            },
            'mask_kwargs': {
                'kernel_shape': cv2.MORPH_RECT,
                'kernel_size': (20, 20),
                'operations': [(cv2.MORPH_CLOSE, None)]
            }
        },
        'REMOVE_ARTIFACTS': {
            'bin_kwargs': {
                'thresh': 'adaptative',
                'size': 7,
                'method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                'c': 2
            },
            'mask_kwargs': {
                'kernel_shape': cv2.MORPH_ELLIPSE,
                'kernel_size': (20, 20),
                'operations': [(cv2.MORPH_CLOSE, None), (cv2.MORPH_OPEN, 2)]
            }
        },
        'FLIP_IMG': {
            'orient': 'left'
        },
        # 'REMOVE_NOISE': {
        #     # 'additive_noise': {
        #     #     'ksize': 3
        #     # },
        #     # 'multiplicative_noise': {
        #     #     'gh': 2,
        #     #     'gl': 0.5,
        #     #     'd0': 30,
        #     #     'c': 1
        #     # },
        # },
        'ECUALIZATION': {
            'clahe_1': {'clip': 1},
            'clahe_2': {'clip': 2}
        },
        'SQUARE_PAD': True,
        'RESIZING': {
            'size': (512, 512)
        }
    }
}


"""
    CARPETAS PRINCIPALES DEL PROGRAMA
"""
RAW_DATA_PATH: io = get_path('..', 'data', '00_RAW')
CONVERTED_DATA_PATH: io = get_path('..', 'data', '01_CONVERTED')
PROCESSED_DATA_PATH: io = get_path('..', 'data', '02_PROCESED')
OUTPUT_DATA_PATH: io = get_path('..', 'data', '03_OUTPUT')
MODEL_DATA_PATH: io = get_path('..', 'models')
LOGGING_DATA_PATH: io = get_path('..', 'logging')

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

        self.model_viz_preprocesing_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'PREPROCESSING')
        self.model_viz_eda_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'DATASET_EDA')
        self.model_viz_train_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'TRAIN_PHASE')
        self.model_viz_data_augm_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'DATA_AUGMENTATION_EXAMPLES')

        self.model_viz_results_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'RESULTS')
        self.model_viz_results_model_history_dir: io = get_path(self.model_viz_results_dir, 'MODEL_HISTORY')
        self.model_viz_results_confusion_matrix_dir: io = get_path(self.model_viz_results_dir, 'CONFUSION_MATRIX')
        self.model_viz_results_accuracy_dir: io = get_path(self.model_viz_results_dir, 'ACCURACY')
        self.model_viz_results_metrics_dir: io = get_path(self.model_viz_results_dir, 'METRICS')



MODEL_FILES = ModelConstants()