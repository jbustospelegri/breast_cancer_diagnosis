import os
import sys
import cv2

from typing import io
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, Affine

from tensorflow.keras.losses import CategoricalCrossentropy
from segmentation_models.metrics import IOUScore
from segmentation_models.losses import DiceLoss, BinaryFocalLoss

from algorithms.metrics import f1_score
from utils.functions import get_path


"""
    CONFIGURACION DEL DATASET
"""
TRAIN_DATA_PROP: float = 0.7

"""
    CONFIGURACION DATA AUGMENTATION
"""
CLASSIFICATION_DATA_AUGMENTATION_FUNCS: dict = {
    'horizontal_flip': HorizontalFlip(),
    'vertical_flip': VerticalFlip(),
    'rotation_90': RandomRotate90(),
    'shift_range': ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.05, interpolation=cv2.INTER_NEAREST),
    'zoom': Affine(scale=[1, 1.1], interpolation=cv2.INTER_LANCZOS4, mode=cv2.BORDER_REPLICATE),
    'shear': Affine(shear=(-10, 10), interpolation=cv2.INTER_LANCZOS4, mode=cv2.BORDER_REPLICATE),
    'rotate': Affine(rotate=(-15, 15), interpolation=cv2.INTER_LANCZOS4, mode=cv2.BORDER_REPLICATE, fit_output=True),
}

SEGMENTATION_DATA_AUGMENTATION_FUNCS: dict = {
    'horizontal_flip': HorizontalFlip(),
    'vertical_flip': VerticalFlip(),
    'zoom': Affine(scale=[1, 1.1], interpolation=cv2.INTER_LANCZOS4, mode=cv2.BORDER_REPLICATE),
    'shear': Affine(shear=(-10, 10), interpolation=cv2.INTER_LANCZOS4, mode=cv2.BORDER_REPLICATE),
}

"""
    CONFIGURACION DE EJECUCIONES DE LOS MODELOS
"""
EPOCHS: int = 150
WARM_UP_EPOCHS: int = 5
LEARNING_RATE: float = 1e-4

SEGMENTATION_BATCH_SIZE: int = 9
BATCH_SIZE: int = 18

# Batch size para fc on top
# BATCH_SIZE: int = 14

SEED: int = 81

CLASSIFICATION_METRICS = {
    'AUC': 'AUC',
    'Precision': 'Precision',
    'Recall': 'Recall',
    'F1 Score': f1_score
}

SEGMENTATION_METRICS = {
    'IoU': IOUScore(),
}

CLASSIFICATION_LOSS = CategoricalCrossentropy()
SEGMENTATION_LOSS = DiceLoss() + (1 * BinaryFocalLoss())

"""
    CONFIGURACION PARA EL MODEL ENSEMBLING
"""
THRESHOLD = 0.5
ENSEMBLER_CONFIG = 'CONF1'
ENSEMBLER_COLS = {
    'CONF1': [],
    'CONF2': ['BREAST', 'BREAST_VIEW', 'BREAST_DENSITY']
}

"""
    CONFIGURACIÓN DE PREPROCESADO DE IMAGENES
"""
IMG_SHAPE: tuple = (384, 192)
PATCH_SIZE: tuple = (300, 300)

CROP_CONFIG: str = 'CONF0'
CROP_PARAMS: dict = {
    'CONF0': {
        'N_BACKGROUND': 1,
        'N_ROI': 2,
        'OVERLAP': 0.9,
        'MARGIN': 1.4
    },
    'CONF1': {
        'N_BACKGROUND': 0,
        'N_ROI': 1,
        'OVERLAP': None,
        'MARGIN': 1.2
    },
    'CONF3': {
        'N_BACKGROUND': 0,
        'N_ROI': 1,
        'OVERLAP': None,
        'MARGIN': 1.4
    }
}

PREPROCESSING_CONFIG: str = 'CONF1'
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
        'RATIO_PAD': {
            'ratio': '1:2',
        },
        'RESIZING': {
            'width': IMG_SHAPE[1],
            'height': IMG_SHAPE[0]
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
TEST_CONVERTED_DATA_PATH: io = get_path(CONVERTED_DATA_PATH, 'Test')

"""
    CARPETAS CON LAS IMAGENES PROCESADAS
"""
CBIS_DDSM_PREPROCESSED_DATA_PATH: io = get_path(PROCESSED_DATA_PATH, PREPROCESSING_CONFIG, 'CBIS_DDSM')
MIAS_PREPROCESSED_DATA_PATH: io = get_path(PROCESSED_DATA_PATH, PREPROCESSING_CONFIG, 'MIAS')
INBREAST_PREPROCESSED_DATA_PATH: io = get_path(PROCESSED_DATA_PATH, PREPROCESSING_CONFIG, 'INBreast')
TEST_PREPROCESSED_DATA_PATH: io = get_path(PROCESSED_DATA_PATH, PREPROCESSING_CONFIG, 'Test')


"""
    CONSTANTES INTERFAZ GRAFICA
"""
APPLICATION_NAME: str = 'Breast Cancer Diagnosis'

"""
    CARPETAS DE LA INTERFAZ GRÁFICA
"""
GUI_CSS_PATH = get_path(WORKING_DIRECTORY, 'static', 'css', create=False)
GUI_HTML_PATH = get_path(WORKING_DIRECTORY, 'static', 'html', 'help_window.html', create=False)
GUI_ICON_PATH = get_path(WORKING_DIRECTORY, 'static', 'images', 'logo.png', create=False)

"""
    CARPETAS DE RESULTADOS DEL MODELO 
"""
DEPLOYMENT_MODELS = get_path(MODEL_DATA_PATH, 'DEPLOYMENT')


class ModelConstants:

    def set_model_name(self, name: str) -> None:
        self.model_root_dir: io = get_path(MODEL_DATA_PATH, name)

        self.model_db_desc_csv: io = get_path(self.model_root_dir, 'dataset.xlsx')
        self.model_db_processing_info_file: io = get_path(self.model_root_dir, 'preprocess_config.json')

        self.model_store_dir: io = get_path(self.model_root_dir, 'STORED_MODELS')
        self.model_store_cnn_dir: io = get_path(self.model_store_dir, 'CNN')
        self.model_store_ensembler_dir: io = get_path(self.model_store_dir, 'MODEL_ENSEMBLING')

        self.model_log_dir: io = get_path(self.model_root_dir, 'TRAIN_LOGS')
        self.model_summary_dir: io = get_path(self.model_root_dir, 'SUMMARY_MODELS')
        self.model_summary_train_csv: io = get_path(self.model_root_dir, 'SUMMARY_MODELS', 'train_summary.csv')

        self.model_predictions_dir: io = get_path(self.model_root_dir, 'PREDICTIONS')
        self.model_predictions_cnn_dir = get_path(self.model_predictions_dir, 'CNN')
        self.model_predictions_ensembler_dir = get_path(self.model_predictions_dir, 'MODEL_ENSEMBLING')

        self.model_data_viz_dir: io = get_path(self.model_root_dir, 'DATA_VIZ')

        self.model_viz_preprocesing_dir: io = get_path(self.model_root_dir, 'PREPROCESSING_VIZ', PREPROCESSING_CONFIG)
        self.model_viz_eda_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'DATASET_EDA')
        self.model_viz_train_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'TRAIN_PHASE')
        self.model_viz_data_augm_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'DATA_AUGMENTATION_EXAMPLES')

        self.model_viz_results_dir: io = get_path(self.model_root_dir, 'DATA_VIZ', 'RESULTS')
        self.model_viz_results_model_history_dir: io = get_path(self.model_viz_results_dir, 'MODEL_HISTORY')
        self.model_viz_results_confusion_matrix_dir: io = get_path(self.model_viz_results_dir, 'CONFUSION_MATRIX')
        self.model_viz_results_auc_curves_dir: io = get_path(self.model_viz_results_dir, 'AUC_CURVES')
        self.model_viz_results_metrics_dir: io = get_path(self.model_viz_results_dir, 'METRICS')


MODEL_FILES = ModelConstants()