import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import dataset_management as dm


# getting .png slices from 3D scans (there was two different datasets)
# inputs = [first dataset path, second dataset path]
# outputs = [first dataset .png imgs, first dataset .png masks,
#            second dataset .png imgs, second dataset .png masks,]
# EXAMPLE
# inputs=['FirstDataset/train/', 'SecondDataset/train/'],
#                                    outputs=['/content/train/data/1/', '/content/train/mask/1/',
#                                             '/content/val/data/1/', '/content/val/mask/1/']

def prepare_datasets_from_3D_scans(inputs, outputs):
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
                name = outputs[0] + 'x%04d%04d.png' % (num, y_index)
                name_mask = outputs[1] + 'x%04d%04d.png' % (num, y_index)
            else:
                name = outputs[2] + 'x%04d%04d.png' % (num, y_index)
                name_mask = outputs[3] + 'x%04d%04d.png' % (num, y_index)
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
                name = outputs[0] + 'x%04d%04d.png' % (num, y_index)
                name_mask = outputs[1] + 'x%04d%04d.png' % (num, y_index)
            else:
                name = outputs[2] + 'x%04d%04d.png' % (num, y_index)
                name_mask = outputs[3] + 'x%04d%04d.png' % (num, y_index)
            plt.imsave(name, data_slice, format='png', cmap='gray', origin='lower')
            plt.imsave(name_mask, mask_slice, format='png', cmap='gray', origin='lower')


if __name__ == '__main__':
    prepare_datasets_from_3D_scans(['firstDataset/train/', 'secondDataset/train/'],
                                   ['/data/train/data/1/', '/data/train/mask/1/',
                                    '/data/val/data/1/', '/data/val/mask/1/'])
