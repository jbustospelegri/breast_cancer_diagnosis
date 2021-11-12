import os
from typing import io

from utils.functions import create_dir


"""
    CONFIGURACION DEL DATASET
"""
TRAIN_DATA_PROP = 0.8
VAL_DATA_PROP = 0.1
TEST_DATA_PROP = 0.1
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

EPOCHS = 100
BATCH_SIZE = 16
SEED = 81

"""
    CARPETAS PRINCIPALES DEL PROGRAMA
"""
TEST_DATA_PATH = os.path.join('..', 'data', '03_TEST')

"""
    CARPETAS CON LOS DATASETS
"""
RAW_DATA_PATH = os.path.join('..', 'data', '00_RAW')
CBIS_DDSM_PATH = os.path.join(RAW_DATA_PATH, 'CBIS-DDSM')
MIAS_PATH = os.path.join(RAW_DATA_PATH, 'MIAS')
INBREAST_PATH = os.path.join(RAW_DATA_PATH, 'INBreast')

"""
    FICHEROS CON LA METADATA DE LAS IMAGENES
"""
CBIS_DDSM_CALC_CASE_DESC_TEST = os.path.join(CBIS_DDSM_PATH, 'calc_case_description_test_set.csv')
CBIS_DDSM_CALC_CASE_DESC_TRAIN = os.path.join(CBIS_DDSM_PATH, 'calc_case_description_train_set.csv')
CBIS_DDSM_MASS_CASE_DESC_TEST = os.path.join(CBIS_DDSM_PATH, 'mass_case_description_test_set.csv')
CBIS_DDSM_MASS_CASE_DESC_TRAIN = os.path.join(CBIS_DDSM_PATH, 'mass_case_description_train_set.csv')
INBREAST_CASE_DESC = os.path.join(INBREAST_PATH, 'INbreast.xls')
MIAS_CASE_DESC = os.path.join(MIAS_PATH, 'Info.txt')

"""
    CARPETAS QUE CONTIENEN LAS IMAGENES
"""
CBIS_DDSM_DB_PATH = os.path.join(CBIS_DDSM_PATH, 'ALL')
MIAS_DB_PATH = os.path.join(MIAS_PATH, 'ALL')
INBREAST_DB_PATH = os.path.join(INBREAST_PATH, 'ALL')

"""
    CARPETAS CON IMAGENES CONVERTIDAS
"""
CONVERTED_DATA_PATH = os.path.join('..', 'data', '01_CONVERTED')
CBIS_DDSM_CONVERTED_DATA_PATH = os.path.join(CONVERTED_DATA_PATH, 'CBIS_DDSM')
MIAS_CONVERTED_DATA_PATH = os.path.join(CONVERTED_DATA_PATH, 'MIAS')
INBREAST_CONVERTED_DATA_PATH = os.path.join(CONVERTED_DATA_PATH, 'INBreast')

"""
    CARPETAS CON LAS IMAGENES TRANSFORMADAS
"""
PROCESSED_DATA_PATH = os.path.join('..', 'data', '02_PROCESED')
CBIS_DDSM_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'CBIS_DDSM')
MIAS_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'MIAS')
INBREAST_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'INBreast')

"""
    CARPETAS DE RESULTADOS 
"""
OUTPUT_DATA_PATH = os.path.join('..', 'data', '03_OUTPUT')
MODEL_DATA_PATH = os.path.join('..', 'models')
LOGGING_DATA_PATH = os.path.join('..', 'logging')
OUTPUT_DATASET_ANALYSIS_DIRPATH = os.path.join(OUTPUT_DATA_PATH, 'DATASET_EDA')


class ModelConstants:

    model_dirname: io = None
    model_results_dirname: io = None
    model_logs_dirname: io = None
    model_summary_dirname: io = None
    model_predictions_dirname: io = None
    model_predictions_filepath: io = None
    model_output_imgs_preproces_dirname: io = None
    model_output_imgs_data_augm_dirname: io = None

    def set_model_name(self, name: str) -> None:
        self.__set_log_dirname(dirname=os.path.join(LOGGING_DATA_PATH, name))
        self.__set_results_dirname(dirname=os.path.join(OUTPUT_DATA_PATH, name, 'RESULTS VISUALIZATION'))
        self.__set_predictions_dirname(dirname=os.path.join(OUTPUT_DATA_PATH, name, 'PREDICTIONS'))
        self.__set_model_dirname(dirname=os.path.join(MODEL_DATA_PATH, name, 'STORED_MODELS'))
        self.__set_summary_dirname(dirname=os.path.join(MODEL_DATA_PATH, name, 'SUMMARY'))
        self.__set_model_preprocessing_dirname(dirname=os.path.join(OUTPUT_DATA_PATH, name, 'PREPROCESSING'))
        self.__set_model_data_augmentation_dirname(dirname=os.path.join(OUTPUT_DATA_PATH, name, 'DATA AUGMENTATION'))

    @create_dir
    def __set_model_dirname(self, dirname: io) -> None:
        self.model_dirname = dirname

    @create_dir
    def __set_results_dirname(self, dirname: io) -> None:
        self.model_results_dirname = dirname

    @create_dir
    def __set_log_dirname(self, dirname: io) -> None:
        self.model_logs_dirname = dirname

    @create_dir
    def __set_predictions_dirname(self, dirname: io) -> None:
        self.model_predictions_dirname = dirname
        self.model_predictions_filepath = os.path.join(dirname, 'predicciones.csv')

    @create_dir
    def __set_summary_dirname(self, dirname: io) -> None:
        self.model_summary_dirname = dirname

    @create_dir
    def __set_model_preprocessing_dirname(self, dirname: io) -> None:
        self.model_output_imgs_preproces_dirname = dirname

    @create_dir
    def __set_model_data_augmentation_dirname(self, dirname: io) -> None:
        self.model_output_imgs_data_augm_dirname = dirname


MODEL_CONSTANTS = ModelConstants()