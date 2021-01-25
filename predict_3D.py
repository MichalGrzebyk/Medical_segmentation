import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from segmentation_models.losses import dice_loss
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score, f1_score
from pathlib import Path
import cv2
import dataset_management as dm
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('first_dataset_path', type=str,
                    help='Path to test subset of first dataset.')
parser.add_argument('second_dataset_path', type=str,
                    help='Path to test subset of second dataset.')
parser.add_argument('model_path', type=str,
                    help='Path to model .md5 file.')
parser.add_argument('predictions_path', type=str,
                    help='Path where predictions will be saved.')


def predict_3D():
    args = parser.parse_args()
    first_dataset_test = args.first_dataset_path
    second_dataset_test = args.second_dataset_path
    predictions_path = args.predictions_path
    model_path = args.model_path
    first_dataset_path = Path(first_dataset_test)
    second_dataset_path = Path(second_dataset_test)
    Path(predictions_path).mkdir(exist_ok=True, parents=True)

    backbone = 'resnet50'
    preprocess_input = get_preprocessing(backbone)

    # define model
    model = Unet(backbone, encoder_weights='imagenet', input_shape=(None, None, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=dice_loss,
                  metrics=[f1_score, iou_score])
    model.load_weights(model_path)

    for scan_path in first_dataset_path.iterdir():
        if scan_path.name.endswith('mask.nii.gz'):
            print(nib.load(str(scan_path)).header.get_zooms())

    print()

    for scan_path in second_dataset_path.iterdir():
        print(nib.load(str(scan_path / 'T1w.nii.gz')).header.get_zooms())

    first_dataset_predictions_path = Path(predictions_path + 'first')
    second_dataset_predictions_path = Path(predictions_path + 'second')

    first_dataset_predictions_path.mkdir(exist_ok=True, parents=True)
    second_dataset_predictions_path.mkdir(exist_ok=True, parents=True)

    fit = MinMaxScaler()

    for scan_path in first_dataset_path.iterdir():
        data, affine = dm.load_raw_volume(scan_path)
        labels = np.zeros(data.shape, dtype=np.uint8)

        x_size, y_size, z_size = data.shape
        for y_index in range(y_size):
            data_slice = data[:, y_index, :]
            data_slice = cv2.resize(data_slice, (256, 256))
            data_slice = fit.fit_transform(data_slice)
            data_slice = cv2.cvtColor(data_slice, cv2.COLOR_GRAY2RGB)
            prediction = model.predict(data_slice[None, :])
            prediction[prediction < 0.5] = 0
            prediction[prediction >= 0.5] = 1
            prediction = prediction.squeeze()
            labels[:, y_index, :] = cv2.resize(prediction, (z_size, x_size))

        dm.save_labels(labels, affine, first_dataset_predictions_path / scan_path.name)

    for scan_path in second_dataset_path.iterdir():
        data, affine = dm.load_raw_volume(scan_path / 'T1w.nii.gz')
        labels = np.zeros(data.shape, dtype=np.uint8)

        x_size, y_size, z_size = data.shape
        for y_index in range(y_size):
            data_slice = data[:, y_index, :]
            data_slice = cv2.resize(data_slice, (256, 256))
            data_slice = fit.fit_transform(data_slice)
            data_slice = cv2.cvtColor(data_slice, cv2.COLOR_GRAY2RGB)
            prediction = model.predict(data_slice[None, :])
            prediction[prediction < 0.5] = 0
            prediction[prediction >= 0.5] = 1
            prediction = prediction.squeeze()
            labels[:, y_index, :] = cv2.resize(prediction, (z_size, x_size))
        dm.save_labels(labels, affine, second_dataset_predictions_path / f'{scan_path.name}.nii.gz')


if __name__ == '__main__':
    predict_3D()
