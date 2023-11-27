import numpy as np
import cv2
import tenforflow as tf
import os, pandas as pd
from sklearn.model_selection import train_test_split
from config import *

bboxes = pd.read_csv(os.path.join(DATASET_ROOT, 'train_ship_segmentations_v2.csv'))

def count_boats(data: pd.Series):
    if data.isnull().any()==True:
        return 0
    else:
        return len(data)

def _get_counts():
    imids = pd.DataFrame(index=np.unique(bboxes['ImageId']))
    imids['Boats'] = bboxes.groupby('ImageId')['EncodedPixels'].apply(count_boats)
    return imids

def decode_mask(row, shapes=IM_SIZE):
    mask = np.zeros((shapes[0] * shapes[1], 1), dtype=np.uint8)

    try:
        ents = row.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (ents[0:][::2], ents[1:][::2])]
        starts -= 1
        ends = starts + lengths

        for wi, hi in zip(starts, ends):
            mask[wi:hi] = 1
    except Exception as e:
        pass
    finally:
        return mask.reshape(shapes).T


def read_sample(imid, ds_root=DATASET_ROOT, subset='train'):
    subset_path = {'train': 'train_v2', 'test': 'test_v2'}
    im = cv2.imread(os.path.join(ds_root, subset_path.get(subset), imid))[..., ::-1]

    if subset == 'train':
        mask = bboxes.set_index('ImageId').loc[imid].values.flatten().tolist()
        mask = np.sum(np.array(list(map(decode_mask, mask))), axis=0)
    else:
        mask = np.zeros((1, 1))

    return im, mask

def tf_dataset_generator(ims):
    for im in ims:
        x, y = data_preprocessing(*read_sample(im, subset='train'))

        yield x, y

def get_dataset(ims, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_generator(
        lambda: tf_dataset_generator(ims),
        output_signature=(
            tf.TensorSpec(shape=(*TARGET_IM_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*TARGET_IM_SIZE, 1), dtype=tf.float32)
        )
    )
    dataset = dataset.map(lambda x, y: {'image': x, 'mask': y})
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def data_preprocessing(image, mask):
    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    mask = tf.reshape(mask, (*tf.shape(mask), 1))

    image = tf.image.convert_image_dtype(image, tf.float32)

    return tf.image.resize(image, TARGET_IM_SIZE), tf.image.resize(mask, TARGET_IM_SIZE)

def preprocess_sample(img_path):
    test_sample = cv2.imread(img_path)[..., ::-1]
    test_sample = cv2.resize(test_sample, TARGET_IM_SIZE)
    test_sample = test_sample / 255.0
    test_sample = test_sample.reshape(1, *TARGET_IM_SIZE, 3)
    return test_sample

def get_train_test(test_size = TEST_SIZE):
    imids = _get_counts()
    train_ims, val_ims = train_test_split(imids,
                                          test_size=test_size,
                                          stratify=imids['Boats'])
    train_ds, val_ds = get_dataset(train_ims.index.tolist()), get_dataset(train_ims.index.tolist())

    return train_ds, val_ds