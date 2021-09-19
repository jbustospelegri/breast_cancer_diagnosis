import os

SEED = 81

CLASS_LABELS = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5'}

RAW_DATA_PATH = os.path.join('..', 'Data', '00_RAW')
PROCESED_DATA_PATH = os.path.join('..', 'Data', '01_PROCESED')
OUTPUT_DATA_PATH = os.path.join('..', 'Data', '02_OUTPUT')

TRAIN_DATA_PATH = os.path.join(RAW_DATA_PATH, 'train')
TEST_DATA_PATH = os.path.join(RAW_DATA_PATH, 'test')
EXAMPLE_IMAGE_DATA_PATH = os.path.join(RAW_DATA_PATH, 'example')

EDA_DATA_PATH = os.path.join(OUTPUT_DATA_PATH, 'EDA')
MODEL_SAVE_DATA_PATH = os.path.join(OUTPUT_DATA_PATH, 'STORED_MODELS')
LOGGING_DATA_PATH = os.path.join(OUTPUT_DATA_PATH, 'LOGS')
MODEL_SUMMARY_DATA_PATH = os.path.join(OUTPUT_DATA_PATH, 'MODEL_SUMMARY')
PREDICTIONS_PATH = os.path.join(OUTPUT_DATA_PATH, 'PREDICTIONS')
VISUALIZATIONS_PATH = os.path.join(OUTPUT_DATA_PATH, 'RESULTS VISUALIZATION')
