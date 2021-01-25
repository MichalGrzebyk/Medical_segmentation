import nibabel as nib
import requests
import zlib
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('first_dataset_predictions_path', type=str,
                    help='Path to predictions made on first dataset.')
parser.add_argument('second_dataset_predictions_path', type=str,
                    help='Path to predictions made on second dataset.')
parser.add_argument('output_file_path', type=str,
                    help='Path where output will be saved as txt file.')


def check_3D():
    args = parser.parse_args()
    first_dataset_predictions = args.first_dataset_predictions_path
    second_dataset_predictions = args.second_dataset_path
    output_file_path = args.output_file_path
    Path(output_file_path).mkdir(exist_ok=True, parents=True)

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
    check_3D()
