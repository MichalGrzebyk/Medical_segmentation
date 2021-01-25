# REMEMBER TO DO: pip install segmentation_models
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from segmentation_models.losses import dice_loss
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score, f1_score
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('model_final_path', type=str,
                    help='Path where final model will be saved.')
parser.add_argument('model_checkpoint_path', type=str,
                    help='Path where model will be saved every epoch.')
parser.add_argument('dataset_path', type=str,
                    help='Path to dataset.')


def from_directory_datagen(path):
    flow_params = {'target_size': (256, 256),
                   'class_mode': None,
                   'color_mode': 'rgb'
                   }

    images_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255,
    )

    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255,
    )

    tr_im = images_datagen.flow_from_directory(
        path + 'train/data/',
        batch_size=32,
        seed=42,
        **flow_params
    )

    tr_mask = mask_datagen.flow_from_directory(
        path + 'train/mask/',
        batch_size=32,
        seed=42,
        **flow_params
    )
    val_im = images_datagen.flow_from_directory(
        path + 'val/data/',
        batch_size=32,
        seed=42,
        **flow_params
    )

    val_mask = mask_datagen.flow_from_directory(
        path + 'val/mask/',
        batch_size=32,
        seed=42,
        **flow_params
    )

    return tr_im, tr_mask, val_im, val_mask


def train_net():
    args = parser.parse_args()
    final_path = args.model_final_path
    checkpoint_path = args.model_checkpoint_path
    dataset_path = args.dataset_path
    with tf.device("/gpu:0"):
        backbone = 'resnet50'
        preprocess_input = get_preprocessing(backbone)

        # load your data
        x_train, y_train, x_val, y_val = from_directory_datagen(dataset_path)

        # preprocess input
        x_train = preprocess_input(x_train)
        x_val = preprocess_input(x_val)

        # define model
        model = Unet(backbone, encoder_weights='imagenet', input_shape=(256, 256, 3))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=dice_loss,
                      metrics=[f1_score, iou_score])

        check_point = [ModelCheckpoint(checkpoint_path + 'model-{epoch:03d}-{val_f1-score:03f}.h5', verbose=1,
                                       monitor='val_f1-score',
                                       save_best_only=True, mode='max')]

        # fit model
        model.fit(
            x=(pair for pair in zip(x_train, y_train)),
            epochs=10,
            steps_per_epoch=x_train.n // x_train.batch_size,
            validation_data=(pair for pair in zip(x_val, y_val)),
            validation_steps=x_val.n // x_val.batch_size,
            verbose=1,
            shuffle=True,
            callbacks=check_point,
        )
        model.save(final_path + 'final_model.h5')


if __name__ == '__main__':
    train_net()
