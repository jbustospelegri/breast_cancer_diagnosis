from typing import io

from breast_cancer_dataset.databases.test_db import DatasetTest
from user_interface.signals_interface import SignalProgressBar, SignalError, SignalCompleted


def generate_predictions_pipeline(excel_filepath: io, signal_information: SignalProgressBar, signal_error: SignalError,
                                  signal_complete: SignalCompleted):
    info = ''
    try:
        info = f'Reading excel {excel_filepath}'
        signal_information.emit_update_label_and_progress_bar(0, info)
        db = DatasetTest(xlsx_io=excel_filepath)

        info = f'Converting images to png format'
        signal_information.emit_update_label_and_progress_bar(5, info)
        db.convert_images_format()

        info = f'ROI extraction'
        signal_information.emit_update_label_and_progress_bar(40, info)
        db.preprocess_image()

        info = ''

        for cnn in search_files(MODEL_DATA_PATH, 'h5'):


            val_augmentations = Compose([
                Lambda(
                    image=lambda x, **kgs: resize_img(x, height=size[0], width=size[1],
                                                      interpolation=cv2.INTER_LANCZOS4),
                    name='image resizing'
                ),
                Lambda(image=lambda x, **kwargs: cast(x, float32), name='floating point conversion'),
                Lambda(image=callback, name='cnn processing function')
            ])

    except Exception as err:
        signal_error.emit_error(__name__, info, err)
    else:
        signal_complete.finish_process()
