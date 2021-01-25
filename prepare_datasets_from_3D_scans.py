import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import dataset_management as dm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('first_dataset_path', type=str,
                    help='Path to first dataset.')
parser.add_argument('second_dataset_path', type=str,
                    help='Path to second dataset.')
parser.add_argument('output_path', type=str,
                    help='Path to directory where data will be saved.')


# getting .png slices from 3D scans (there was two different datasets)

def prepare_datasets_from_3D_scans():
    args = parser.parse_args()
    inputs = [args.first_dataset_path, args.second_dataset_path]
    outputs = [args.output_path + 'train/', args.output_path + 'val/']
    Path(outputs[0] + 'data').mkdir(exist_ok=True, parents=True)
    Path(outputs[1] + 'data').mkdir(exist_ok=True, parents=True)
    Path(outputs[0] + 'mask').mkdir(exist_ok=True, parents=True)
    Path(outputs[1] + 'mask').mkdir(exist_ok=True, parents=True)

    data_path = [name for name in sorted(Path(inputs[1]).iterdir())]
    size = len(data_path)
    for num, file in enumerate(data_path):
        data_path2 = [name_ for name_ in sorted(Path(file).iterdir())]
        d_p = data_path2[0]
        m_p = data_path2[1]
        tmp_img, aff = dm.load_raw_volume(d_p)
        tmp_musk = dm.load_labels_volume(m_p)
        x_size, y_size, z_size = tmp_img.shape
        for y_index in range(y_size):
            data_slice = tmp_img[:, y_index]
            data_slice = cv2.resize(data_slice, (256, 256))
            mask_slice = tmp_musk[:, y_index]
            mask_slice = cv2.resize(mask_slice, (256, 256))
            if num / size < 0.9:
                name = outputs[0] + 'data/x%04d%04d.png' % (num, y_index)
                name_mask = outputs[0] + 'mask/x%04d%04d.png' % (num, y_index)
            else:
                name = outputs[1] + 'data/x%04d%04d.png' % (num, y_index)
                name_mask = outputs[1] + 'mask/x%04d%04d.png' % (num, y_index)
            plt.imsave(name, data_slice, format='png', cmap='gray', origin='lower')
            plt.imsave(name_mask, mask_slice, format='png', cmap='gray', origin='lower')

    data_path = [name for name in sorted(Path(inputs[0]).iterdir()) if not name.name.endswith('mask.nii.gz')]
    size = len(data_path)
    for num, file in enumerate(data_path):
        tmp_img, aff = dm.load_raw_volume(file)
        tmp_musk = dm.load_labels_volume(str(file).replace(".nii.gz", "_mask.nii.gz"))
        x_size, y_size, z_size = tmp_img.shape
        for y_index in range(y_size):
            data_slice = tmp_img[:, y_index]
            data_slice = cv2.resize(data_slice, (256, 256))
            mask_slice = tmp_musk[:, y_index]
            mask_slice = cv2.resize(mask_slice, (256, 256))
            if num / size < 0.9:
                name = outputs[0] + 'data/x%04d%04d.png' % (num, y_index)
                name_mask = outputs[0] + 'mask/x%04d%04d.png' % (num, y_index)
            else:
                name = outputs[1] + 'data/x%04d%04d.png' % (num, y_index)
                name_mask = outputs[1] + 'mask/x%04d%04d.png' % (num, y_index)
            plt.imsave(name, data_slice, format='png', cmap='gray', origin='lower')
            plt.imsave(name_mask, mask_slice, format='png', cmap='gray', origin='lower')


if __name__ == '__main__':
    prepare_datasets_from_3D_scans()
