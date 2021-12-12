import io
import os
import cv2
import plistlib

import numpy as np
import pydicom

from skimage.draw import polygon
from PIL import Image

from src.utils.config import INBREAST_DB_XML_ROI_PATH, LOGGING_DATA_PATH, CBIS_DDSM_DB_PATH
from src.utils.functions import load_point, get_path, get_filename, search_files


def get_inbreast_roi_mask(args) -> None:
    try:
        if not len(args) == 3:
            raise ValueError('Incorrect number of arguments for function get_inbrest_roi_mask')

        ori_img = args[0]
        xlsm_file = args[1]
        out_file = args[2]

        if not os.path.isfile(ori_img):
            raise FileNotFoundError(f'{ori_img} image does not exists.')

        if not os.path.isfile(get_path(INBREAST_DB_XML_ROI_PATH, f'{xlsm_file}.xml')):
            raise FileNotFoundError(f'{xlsm_file} xlsm does not exists.')

        assert not os.path.isfile(out_file), f'Mask {out_file} already created.'
        shape = cv2.imread(ori_img).shape

        mask = np.zeros(shape)

        with open(get_path(INBREAST_DB_XML_ROI_PATH, f'{xlsm_file}.xml'), 'rb') as f:
            plist_dict = plistlib.load(f, fmt=plistlib.FMT_XML)['Images'][0]
            for roi in plist_dict['ROIs']:
                if roi['Name'] in ['Mass']:
                    x, y = zip(*[load_point(point) for point in roi['Point_px']])
                    poly_x, poly_y = polygon(np.array(x), np.array(y), shape=shape)
                    mask[poly_x, poly_y] = 255

        Image.fromarray(np.uint8(mask)).save(out_file)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors (Assertions).txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except ValueError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nError calling function get_inbrest_roi_mask pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(ori_img)}\n{err}\n{"=" * 100}')


def get_mias_roi_mask(args) -> None:

    try:
        if len(args) != 4:
            raise ValueError('Incorrect number of args for function get_mias_roi')

        assert not os.path.isfile(args[0]), f'Mask {args[0]} already created.'

        mask = np.zeros(shape=(1024, 1024), dtype=np.uint8)
        for x, y, rad in zip(args[1], args[2], args[3]):
            cv2.circle(mask, center=(int(x), int(y)), radius=int(rad), thickness=-1, color=(255, 255, 255))
        cv2.imwrite(args[0], mask)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors (Assertions).txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except ValueError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nError calling function get_inbrest_roi_mask pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(get_filename(args[0]))}\n{err}\n{"=" * 100}')


def get_cbis_roi_mask(args) -> None:

    try:
        if len(args) != 2:
            raise ValueError('Incorrect number of arguments for function get_cbis_roi_mask')

        assert not os.path.isfile(args[1]), f'Mask {args[1]} already created.'

        masks = []
        for img in search_files(file=get_path(CBIS_DDSM_DB_PATH, f'{args[0]}*_[0-9]'), ext='dcm'):

            # Se lee la información de las imagenes en formato dcm
            img = pydicom.dcmread(img)

            # Se convierte las imagenes a formato de array
            img_array = img.pixel_array.astype(float)

            if len(np.unique(img_array)) == 2:

                # Se realiza un reescalado de la imagen para obtener los valores entre 0 y 255
                rescaled_image = (np.maximum(img_array, 0) / img_array.max()) * 255

                # Se limpian mascaras que sean de menos de 5 píxeles
                _, _, h, w = cv2.boundingRect(np.uint8(rescaled_image))

                if (h > 10) and (w > 10):
                    # Se convierte la imagen al ipode datos unsigned de 8 bytes
                    masks.append(np.uint8(rescaled_image))

        final_mask = sum(masks)
        final_mask[final_mask > 1] = 255

        cv2.imwrite(args[1], final_mask)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors (Assertions).txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except ValueError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nError calling function get_cbis_roi_mask pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(args[1])}\n{err}\n{"=" * 100}')