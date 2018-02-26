import os
import glob
import numpy as np
import cv2
import random
import shutil
import Augmentor
from sklearn.utils import shuffle

def class_list(data_path):
    classes = [c for c in os.listdir(data_path) if not os.path.isfile(os.path.join(data_path,c))]
    return classes

def make_folders(data_path):
    files_full = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path,f))]
    files_name = [os.path.splitext(files_full[i])[0] for i in range(len(files_full))]
    
    for m, n in zip(files_full, files_name):
        src = os.path.join(data_path,m)
        des = os.path.join(data_path,n,m)
        des_dir = os.path.join(data_path,n)
        if not os.path.exists(des_dir):
            os.mkdir(des_dir)    
        shutil.copyfile(src,des)
    
def augment_data(path, no_of_aug_samples):   # class path
    p = Augmentor.Pipeline(path)
    
    p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)
    p.zoom(probability=0.5, min_factor=1.0, max_factor=1.1)
    p.skew(probability=0.5, magnitude=0.1)
    p.random_distortion(probability=0.5, grid_height=2, grid_width=2, magnitude=1)
    p.shear(probability=0.5, max_shear_left=5, max_shear_right=5)
    p.sample(no_of_aug_samples)
    
    
def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl,0)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def load_test(test_path, image_size, classes):
    X_test = []
    X_test_id = []
    print("Reading test images")
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(test_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl,0)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            flbase = os.path.basename(fl)
            X_test.append(image)
            X_test_id.append(flbase)

    ### because we're not creating a DataSet object for the test images, normalization happens here
    X_test = np.array(X_test, dtype=np.uint8)
    X_test = X_test.astype('float32')
    X_test = X_test / 255

    return X_test, X_test_id

class DataSet(object):
    def __init__(self, images, labels, ids, cls):
        """Construct a DataSet. one_hot arg is used only if fake_data is true."""

        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # Convert from [0, 255] -> [0.0, 1.0].

        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def ids(self):
        return self._ids

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # # Shuffle the data (maybe)
            # perm = np.arange(self._num_examples)
            # np.random.shuffle(perm)
            # self._images = self._images[perm]
            # self._labels = self._labels[perm]
            # Start next epoch

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]

def read_train_sets(train_path, image_size, classes, validation_size, seed):
    class DataSets(object):
        pass
    data_sets = DataSets()

    images, labels, ids, cls = load_train(train_path, image_size, classes)
    images, labels, ids, cls = shuffle(images, labels, ids, cls, random_state=seed)  # shuffle the data

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_ids = ids[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

    return data_sets


def read_test_set(test_path, image_size, classes, seed):
    images, ids  = load_test(test_path, image_size, classes)
    images, ids = shuffle(images, ids, random_state=seed)
    return images, ids

def split_data_set(data_path, train_path, test_path, classes, validation_size, test_size, seed):
    print('Splitting data set')

    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        print('Creating {} files'.format(fld))
        path = os.path.join(data_path, fld, 'output')
        path_train = os.path.join(train_path, fld)
        path_test = os.path.join(test_path, fld)
        filenames = os.listdir(path)

        random.seed(seed)
        filenames.sort()
        random.shuffle(filenames)

        split = int(test_size * len(filenames))
        train_filenames = filenames[split:]         # dev set is inside train and is taken care of later
        test_filenames = filenames[:split]

        for paths in [path_train, path_test]:
            if not os.path.exists(paths):
                os.mkdir(paths)
            else:
                cmd = input("Warning: output dir {} already exists. Continue? (y/n)".format(paths))
                if cmd != "y":
                    return

        # copy and move to train and test folders
        for name in train_filenames:
            src = os.path.join(path,name)
            des = os.path.join(path_train,name)
            shutil.copyfile(src,des)
        for name in test_filenames:
            src = os.path.join(path,name)
            des = os.path.join(path_test,name)
            shutil.copyfile(src,des)

        print("Done building dataset")