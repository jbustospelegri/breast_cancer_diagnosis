from utils.functions import get_path


"""
    CONFIGURACION DEL DATASET
"""
TRAIN_DATA_PROP = 0.8
VAL_DATA_PROP = 0.1
TEST_DATA_PROP = 0.1

"""
    CONFIGURACION DATA AUGMENTATION
"""

DATA_AUGMENTATION_FUNCS = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'shear_range': 0.1,
    'zoom_range': 0.2,
    # 'rotation_range': 40,
}

"""
    CONFIGURACION DE EJECUCIONES DE LOS MODELOS
"""

EPOCHS: int = 100
WARM_UP_EPOCHS: int = 30
LEARNING_RATE: float = 1e-4
WARM_UP_LEARNING_RATE: float = 1e-3

BATCH_SIZE = 16
SEED = 81

"""
    CONFIGURACIÓN DE PREPROCESADO DE IMAGENES
"""
PREPROCESSING_CONFIG = 'CONF1'
PREPROCESSING_FUNCS = {
    'CONF1': {
        'CROPPING_1': {
            'left': 0.01,
            'right': 0.01,
            'top': 0.04,
            'bottom': 0.04
        },
        'MIN_MAX_NORM': True,
        'REMOVE_ARTIFACTS': {
            'bin_kwargs': {
                'thresh': 30,
                'maxval': 1,
            },
            'mask_kwargs': {
                'kernel_size': (60, 60)
            }
        },
        'FLIP_IMG': {
            'orient': 'left'
        },
        'ECUALIZATION': {
            'clahe_1': {'clip': 1},
            'clahe_2': {'clip': 2}
        },
        'CROPPING_2': {
            'left': 0.05,
            'right': 0,
            'top': 0,
            'bottom': 0
        },
        'SQUARE_PAD': True,
        'RESIZING': {
            'size': (1024, 1024)
        }
    },
    'CONF2': {
        'CROPPING_1': {
            'left': 0.01,
            'right': 0.01,
            'top': 0.04,
            'bottom': 0.04
        },
        'MIN_MAX_NORM': True,
        'REMOVE_ARTIFACTS': {
            'bin_kwargs': {
                'thresh': 30,
                'maxval': 1,
            },
            'mask_kwargs': {
                'kernel_size': (60, 60)
            }
        },
        'FLIP_IMG': {
            'orient': 'left'
        },
        'ECUALIZATION': {
            'clahe_1': {
                'clip': 1
            }
        },
        'CROPPING_2': {
            'left': 0.05,
            'right': 0,
            'top': 0,
            'bottom': 0
        },
        'SQUARE_PAD': True,
        'RESIZING': {
            'size': (1024, 1024)
        }
    }
}


"""
    CARPETAS PRINCIPALES DEL PROGRAMA
"""
RAW_DATA_PATH = get_path('..', 'data', '00_RAW')
CONVERTED_DATA_PATH = get_path('..', 'data', '01_CONVERTED')
PROCESSED_DATA_PATH = get_path('..', 'data', '02_PROCESED')
OUTPUT_DATA_PATH = get_path('..', 'data', '03_OUTPUT')
MODEL_DATA_PATH = get_path('..', 'models')
LOGGING_DATA_PATH = get_path('..', 'logging')

"""
    CARPETAS CON LOS DATASETS
"""
CBIS_DDSM_PATH = get_path(RAW_DATA_PATH, 'CBIS-DDSM')
MIAS_PATH = get_path(RAW_DATA_PATH, 'MIAS')
INBREAST_PATH = get_path(RAW_DATA_PATH, 'INBreast')

"""
    FICHEROS CON LA METADATA DE LAS IMAGENES
"""
CBIS_DDSM_CALC_CASE_DESC_TEST = get_path(CBIS_DDSM_PATH, 'calc_case_description_test_set.csv')
CBIS_DDSM_CALC_CASE_DESC_TRAIN = get_path(CBIS_DDSM_PATH, 'calc_case_description_train_set.csv')
CBIS_DDSM_MASS_CASE_DESC_TEST = get_path(CBIS_DDSM_PATH, 'mass_case_description_test_set.csv')
CBIS_DDSM_MASS_CASE_DESC_TRAIN = get_path(CBIS_DDSM_PATH, 'mass_case_description_train_set.csv')
INBREAST_CASE_DESC = get_path(INBREAST_PATH, 'INbreast.xls')
MIAS_CASE_DESC = get_path(MIAS_PATH, 'Info.txt')

"""
    CARPETAS QUE CONTIENEN LAS IMAGENES
"""
CBIS_DDSM_DB_PATH = get_path(CBIS_DDSM_PATH, 'ALL')
MIAS_DB_PATH = get_path(MIAS_PATH, 'ALL')
INBREAST_DB_PATH = get_path(INBREAST_PATH, 'ALL')

"""
    CARPETAS CON IMAGENES CONVERTIDAS
"""
CBIS_DDSM_CONVERTED_DATA_PATH = get_path(CONVERTED_DATA_PATH, 'CBIS_DDSM')
MIAS_CONVERTED_DATA_PATH = get_path(CONVERTED_DATA_PATH, 'MIAS')
INBREAST_CONVERTED_DATA_PATH = get_path(CONVERTED_DATA_PATH, 'INBreast')

"""
    CARPETAS CON LAS IMAGENES PROCESADAS
"""


class PreprocesConstants:

    def set_preproces_name(self, name: str) -> None:
        self.preproces_root_dir = get_path(PROCESSED_DATA_PATH, name)
        self.cbis_ddsm_processed_dir = get_path(self.preproces_root_dir, 'CBIS_DDSM')
        self.mias_processed_dir = get_path(self.preproces_root_dir, 'MIAS')
        self.inbreast_processed_dir = get_path(self.preproces_root_dir, 'INBreast')


PREPROCES_CONSTANTS = PreprocesConstants()

"""
    CARPETAS DE RESULTADOS DEL MODELO 
"""


class ModelConstants:

    def set_model_name(self, name: str) -> None:
        self.model_root_dir = get_path(MODEL_DATA_PATH, name)

        self.model_db_dir = get_path(self.model_root_dir, 'DATASET')
        self.model_db_procesing_dir = get_path(self.model_root_dir, 'DATASET', 'INPUT_IMGS_PREPROCESING')
        self.model_db_data_augm_dir = get_path(self.model_root_dir, 'DATASET', 'DATA_AUGMENTATION_EXAMPLES')
        self.model_db_eda_dir = get_path(self.model_root_dir, 'DATASET', 'DATASET_EDA')
        self.model_db_desc_csv = get_path(self.model_root_dir, 'DATASET', 'dataset.xlsx')

        self.model_store_dir = get_path(self.model_root_dir, 'STORED_MODELS')
        self.model_log_dir = get_path(self.model_root_dir, 'TRAIN_LOGS')
        self.model_summary_dir = get_path(self.model_root_dir, 'SUMMARY_MODELS')

        self.model_data_viz_dir = get_path(self.model_root_dir, 'DATA_VIZ')
        self.model_viz_train_dir = get_path(self.model_root_dir, 'DATA_VIZ', 'TRAIN_PHASE')
        self.model_viz_results_dir = get_path(self.model_root_dir, 'DATA_VIZ', 'RESULTS')


MODEL_CONSTANTS = ModelConstants()