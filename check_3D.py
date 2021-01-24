import nibabel as nib
import requests
import zlib
from pathlib import Path


def check_3D(first_dataset_predictions, second_dataset_predictions, output_file_path):
    mean = 0
    i = 0
    first_dataset_predictions_path = Path(first_dataset_predictions)
    second_dataset_predictions_path = Path(second_dataset_predictions)
    for dataset_predictions_path in (first_dataset_predictions_path, second_dataset_predictions_path):
        for prediction_path in dataset_predictions_path.iterdir():
            prediction_name = prediction_path.name[:-7]  # deleting '.nii.gz' from filename
            prediction = nib.load(str(prediction_path))

            # predictions were checked on server started and controlled by the teacher
            response = requests.post(f'link to prediction checker{prediction_name}',
                                     data=zlib.compress(prediction.to_bytes()))
            if response.status_code == 200:
                print(dataset_predictions_path.name, prediction_path.name, response.json())
                mean += response.json()['dice']
                i += 1
            else:
                print(f'Error processing prediction {dataset_predictions_path.name}/{prediction_name}: {response.text}')

    f = open(output_file_path + 'score.txt', 'w')
    f.write('Score = ' + str(mean/i))


if __name__ == '__main__':
    check_3D('/predictions/first/', '/predictions/second/', '/predictions/')
