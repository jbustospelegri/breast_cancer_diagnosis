import cv2
import pandas as pd

from typing import io
from segmentation_models import get_preprocessing
from tensorflow.keras.models import load_model

from breast_cancer_dataset.databases.test_db import DatasetTest
from cnns.metrics import f1_score
from cnns.utils import get_predictions
from cnns.model_ensambling import GradientBoosting
from user_interface.signals_interface import SignalProgressBar, SignalError, SignalCompleted, SignalLogging
from utils.config import DEPLOYMENT_MODELS
from utils.functions import get_filename, get_path, get_contours


def generate_predictions_pipeline(
        excel_filepath: io, out_dirpath: io, signal_information: SignalProgressBar, signal_error: SignalError,
        signal_complete: SignalCompleted, signal_log: SignalLogging
):
    info = ''
    predictions = {}
    cnn_models = {
        'densenet121':  get_path(DEPLOYMENT_MODELS, 'DenseNet.h5'),
        'inceptionv3': get_path(DEPLOYMENT_MODELS, 'InceptionV3.h5'),
        'resnet50': get_path(DEPLOYMENT_MODELS, 'ResNet50.h5'),
        'vgg16': get_path(DEPLOYMENT_MODELS, 'VGG16.h5'),
    }
    ensambling_model = get_path(DEPLOYMENT_MODELS, 'GradientBoosting.sav')

    try:
        info = f'Reading excel file'
        signal_information.emit_update_label_and_progress_bar(0, info)
        db = DatasetTest(xlsx_io=excel_filepath, signal=signal_log, out_path=out_dirpath)

        info = f'Converting images to png format'
        signal_information.emit_update_label_and_progress_bar(2, info)
        db.convert_images_format(signal_information, min_value=2, max_value=40)

        info = f'ROI extraction'
        signal_information.emit_update_label_and_progress_bar(40, info)
        db.preprocess_image(signal=signal_information, min_value=40, max_value=65)

        info = 'Generating ROI classification'
        signal_information.emit_update_label_and_progress_bar(68, info)
        for i, (model_name, path) in enumerate(cnn_models.items(), 1):

            model = load_model(path, custom_objects={'f1_score': f1_score})

            data = db.get_iterator(get_preprocessing(model_name), size=model.input_shape[1:3])

            predictions[get_filename(path)] = get_predictions(keras_model=model, data=data)

            signal_information.emit_update_label_and_progress_bar(68 + i * 3, info)

        model_ensambler = GradientBoosting(db=data, model_path=ensambling_model)

        final_data = pd.merge(
            left=db.df,
            right=model_ensambler.predict(data, **predictions),
            on=['PROCESSED_IMG'],
            how='left'
        )

        info = 'Bulking results'
        signal_information.emit_update_label_and_progress_bar(80, info)

        final_data.to_excel(get_path(out_dirpath, f'{get_filename(excel_filepath)}.xlsx'), index=False)
        for i, (idx, row) in enumerate(final_data.iterrows(), 1):

            try:

                bar_range = 100 - 80
                img = cv2.imread(row.CONVERTED_IMG, cv2.IMREAD_COLOR)
                x, y, w, h = cv2.boundingRect(get_contours(cv2.imread(row.CONVERTED_MASK, cv2.IMREAD_GRAYSCALE))[0])

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=5)
                cv2.imwrite(get_path(out_dirpath, row.PATHOLOGY, f'{row.ID}.png'), img)

            except Exception as err:
                with open(get_path(out_dirpath, f'Preprocessing Errors.txt'), 'a') as f:
                    f.write(f'{"=" * 100}\n{row.ID}\n{err}\n{"=" * 100}')

            signal_information.emit_update_label_and_progress_bar(
               80 + bar_range * i // len(final_data), f'Saved image {i} of {len(final_data)}'
            )

        # Recuperar información de los logs de errores.


    except Exception as err:
        signal_error.emit_error(__name__, info, err, True)
    else:
        signal_complete.finish_process()
