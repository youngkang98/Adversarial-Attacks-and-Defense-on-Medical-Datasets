import torch.utils.data as data
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv
import random
import numpy as np
import pandas as pd
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Optional heavy dependencies — only needed for specific dataset loaders
try:
    import kornia.augmentation as A
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import to_categorical
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# COVIDx8B uses 'negative'/'positive'; older splits use 'normal'/'pneumonia'
mapping = {
    'negative': 0, 'normal': 0,
    'positive': 1, 'pneumonia': 1,
    'COVID-19': 2,
}
classes = ['melanoma', 'seborrheic keratosis', 'nevus', 'basal cell carcinoma', 'squamous cell carcinoma', 'dermatofibroma', 'vascular lesion']

from torch.utils.data import Dataset

class ISIC2018Dataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        documentDir = os.path.expanduser('~\Documents')
        imageDir = os.path.join(documentDir, self.df['path'][idx])
        image = Image.open(imageDir).convert("RGB")
        label = torch.tensor(int(self.df['label_idx'][idx]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class ISICDataset(Dataset):
    def __init__(self, datapath, csv_file, data_type, transform=None, exclude_first_n=None, one_hot_encode=False, num_classes=None):
        self.datapath = datapath
        self.csv_file = csv_file
        self.transform = transform
        self.data_type = data_type
        self.data = []
        self.labels = []
        self.one_hot_encode = one_hot_encode
        self.num_classes = num_classes
        
        # Load data and labels from CSV file or pre-processed files
        self._load_data()

        # If exclude_first_n is provided, exclude the first n images
        if exclude_first_n is not None:
            self.data = self.data[exclude_first_n:]
            self.labels = self.labels[exclude_first_n:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Check if idx is a slice
        if isinstance(idx, slice):
            images, labels = [], []
            for i in range(*idx.indices(len(self))):
                img, label = self.get_item(i)
                images.append(img)
                labels.append(label)
            
            # Convert lists to numpy arrays
            images_np = np.array([np.array(img) for img in images])  # Ensure images are numpy arrays
            labels_np = np.array(labels)
            return images_np, labels_np
        else:
            return self.get_item(idx)
    
    def get_item(self, idx):
        img_name = os.path.join(self.datapath, self.data[idx])
        image = Image.open(img_name).convert("RGB")
        label = self.labels[idx]
        
        # Check if one-hot encoding is needed
        if self.one_hot_encode and self.num_classes is not None:
            one_hot_label = self._one_hot_encode_label(label)
        else:
            one_hot_label = label
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Return one-hot encoded label if needed
        return image, one_hot_label if self.one_hot_encode else label
    
    def _one_hot_encode_label(self, label):
        one_hot = np.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot
        
    def _load_data(self):
        if os.path.exists(self.datapath + '/x_{}_n.npy'.format(self.data_type)):
            self.data = np.load(self.datapath + '/x_{}.npy'.format(self.data_type))
            self.labels = np.load(self.datapath + '/y_{}.npy'.format(self.data_type))
        else:
            with open(self.csv_file, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the header row if present
                count = 0
                for row in csv_reader:
                    img_name = row[0] + '.jpg'  # Assuming image extension is ".jpg"
                    label = int(row[1])
                    
                    self.data.append(img_name)
                    self.labels.append(label)
                    
                    count += 1
                    # if count == 1000:
                    #     break
            
            # Save the pre-processed data to disk
            np.save('{}/x_{}'.format(self.datapath, self.data_type), np.array(self.data))
            np.save('{}/y_{}'.format(self.datapath, self.data_type), np.array(self.labels))

class ISICDatasetTF:
    def __init__(self, datapath, csv_file, data_type):
        self.datapath = datapath
        self.csv_file = csv_file
        self.data_type = data_type
        self.data = []
        self.labels = []
        
        # Load data and labels from CSV file or pre-processed files
        self._load_data()

    def _load_data(self):
        if os.path.exists(os.path.join(self.datapath, 'x_{}_n.npy'.format(self.data_type))):
            self.data = np.load(os.path.join(self.datapath, 'x_{}.npy'.format(self.data_type)))
            self.labels = np.load(os.path.join(self.datapath, 'y_{}.npy'.format(self.data_type)))
        else:
            with open(self.csv_file, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the header row if present
                for row in csv_reader:
                    img_name = row[0] + '.jpg'  # Assuming image extension is ".jpg"
                    label = int(row[1])
                    
                    self.data.append(img_name)
                    self.labels.append(label)
            
            # Save the pre-processed data to disk
            np.save(os.path.join(self.datapath, 'x_{}'.format(self.data_type)), np.array(self.data))
            np.save(os.path.join(self.datapath, 'y_{}'.format(self.data_type)), np.array(self.labels))

    def parse_function(self, filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [224, 224])
        image_normalized = image_resized / 255.0
        one_hot_label = tf.one_hot(label, depth=8)
        return image_normalized, one_hot_label


    def create_tf_dataset(self):
        filenames = [os.path.join(self.datapath, name) for name in self.data]
        labels = self.labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

class DatasetSeprateByClass(Dataset):
    def __init__(self, root_dir, csv_file, data_type, transform=None, exclude_first_n=None, one_hot_encode=False, num_classes=None):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.data_type = data_type
        self.data = []
        self.labels = []
        self.one_hot_encode = one_hot_encode
        self.num_classes = num_classes
        
        # Define the mapping from class names to integers
        self.label_mapping = self._create_label_mapping()

        # Load data and labels from CSV file or pre-processed files
        self._load_data()

        # If exclude_first_n is provided, exclude the first n images
        if exclude_first_n is not None:
            self.data = self.data[exclude_first_n:]
            self.labels = self.labels[exclude_first_n:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Check if idx is a slice
        if isinstance(idx, slice):
            images, labels = [], []
            for i in range(*idx.indices(len(self))):
                img, label = self.get_item(i)
                images.append(img)
                labels.append(label)
            
            # Convert lists to numpy arrays
            images_np = np.array([np.array(img) for img in images])  # Ensure images are numpy arrays
            labels_np = np.array(labels)
            return images_np, labels_np
        else:
            return self.get_item(idx)
    
    def get_item(self, idx):
        img_name = self.data[idx]
        class_name = self.labels[idx]
        
        # Dynamically construct the full path to the image
        img_path = os.path.join(self.root_dir, class_name, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Check if one-hot encoding is needed
        if self.one_hot_encode and self.num_classes is not None:
            one_hot_label = self._one_hot_encode_label(class_name)
        else:
            one_hot_label = self.label_mapping[class_name]
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Return one-hot encoded label if needed
        return image, one_hot_label if self.one_hot_encode else self.label_mapping[class_name]
    
    def _one_hot_encode_label(self, label):
        one_hot = np.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot

    def _create_label_mapping(self):
        # Create a mapping from class names (strings) to integers
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        label_mapping = {cls: idx for idx, cls in enumerate(classes)}
        return label_mapping
        
    def _load_data(self):
        with open(self.csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            # Read the image extension from the first row
            self.img_extension = next(csv_reader)[0]
            for row in csv_reader:
                img_name = row[0]
                class_name = row[1]
                
                self.data.append(img_name)
                self.labels.append(class_name)
        
        # Save the pre-processed data to disk
        np.save('{}/x_{}'.format(self.root_dir, self.data_type), np.array(self.data))
        np.save('{}/y_{}'.format(self.root_dir, self.data_type), np.array(self.labels))


# Define the OCTDataset class
class OCTDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.datapath = datapath
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Load data from folders
        self._load_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.datapath, self.data[idx])
        image = Image.open(img_name).convert("RGB")
        label = self.labels[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def _load_data(self):
        class_folders = [d for d in os.listdir(self.datapath) if os.path.isdir(os.path.join(self.datapath, d))]
        class_folders.sort()
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}
        
        for class_name in class_folders:
            class_dir = os.path.join(self.datapath, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    image_path = os.path.join(class_name, img_name)
                    self.data.append(image_path)
                    self.labels.append(class_to_idx[class_name])
                    

from torch.utils.data import Dataset

class COVID19Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, exclude_first_n=None, one_hot_encode=False, num_classes=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.one_hot_encode = one_hot_encode
        self.num_classes = num_classes
        
        # If exclude_first_n is provided, exclude the first n images
        if exclude_first_n is not None:
            self.data = self.data.iloc[exclude_first_n:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            images, labels = [], []
            for i in range(*idx.indices(len(self))):
                img, label = self.get_item(i)
                images.append(img)
                labels.append(label)
            images_np = np.array([np.array(img) for img in images])
            labels_np = np.array(labels)
            return images_np, labels_np
        else:
            return self.get_item(idx)
    
    def get_item(self, idx):
        row = self.data.iloc[idx]
        img_name = row['image_name']
        label = row['label']
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.one_hot_encode and self.num_classes is not None:
            label = self._one_hot_encode_label(label)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _one_hot_encode_label(self, label):
        one_hot = np.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot


class COVID19Dataset_New(Dataset):
    """
    Reads the COVIDx .txt split format used by COVID-Net:
        patient_id  filename  label  split
    where label is one of: normal, pneumonia, COVID-19
    Images are expected flat in root_dir (no class subfolders).
    """
    def __init__(self, txt_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self._load(txt_file)

    def _load(self, txt_file):
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                filename = parts[1]
                label_str = parts[2]
                self.data.append(filename)
                self.labels.append(mapping.get(label_str, 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            imgs, lbls = [], []
            for i in range(*idx.indices(len(self))):
                img, lbl = self._get_item(i)
                imgs.append(img)
                lbls.append(lbl)
            return np.array([np.array(im) for im in imgs]), np.array(lbls)
        return self._get_item(idx)

    def _get_item(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def load_data(datapath, trainfile, testfile):
    if os.path.exists(datapath + '/x_train.npy'):
        x_train = np.load(datapath + '/x_train.npy')
        y_train = np.load(datapath + '/y_train.npy')
        x_test = np.load(datapath + '/x_test.npy')
        y_test = np.load(datapath + '/y_test.npy')
    else:
        files = {'train': trainfile, 'test': testfile}
        dataset = {}
        for data_type in ['train', 'test']:
            print('make {} dataset'.format(data_type))
            x_list = []
            y_list = []
            datafile = open(files[data_type], 'r')
            datafile = datafile.readlines()
            for i in tqdm(range(1,len(datafile))):  # len(datafile))):
                line = datafile[i].split(",")
                x = cv2.imread(os.path.join(datapath, line[0]+'.jpg'))
                h, w, c = x.shape
                x = x[int(h / 6):, :]
                x = cv2.resize(x, (224, 224))
                x = x.astype('float32') / 255.0
                x = np.transpose(x, (2, 0, 1))  # Transpose the image array
                x_list.append(x)
                y_list.append(line[1])
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            num_cls = len(set(int(v) for v in y_list))
            y_list = np.eye(num_cls)[np.array(y_list, dtype=int)]
            np.save('{}/x_{}'.format(datapath, data_type), x_list)
            np.save('{}/y_{}'.format(datapath, data_type), y_list)
            dataset['x_{}'.format(data_type)] = x_list
            dataset['y_{}'.format(data_type)] = y_list
        x_train = dataset['x_train']
        y_train = dataset['y_train']
        x_test = dataset['x_test']
        y_test = dataset['y_test']
    mean_l2_train = 0
    mean_inf_train = 0
    for im in x_train:
        mean_l2_train += np.linalg.norm(im[:, :, 0].flatten(), ord=2)
        mean_inf_train += np.abs(im[:, :, 0].flatten()).max()
    mean_l2_train /= len(x_train)
    mean_inf_train /= len(x_train)
    min_, max_ = float(np.amin(x_train)), float(np.amax(x_train))
    return (x_train, y_train), (x_test, y_test), min_, max_, mean_l2_train, mean_inf_train

def load_data_without_train(datapath, testfile):
    if os.path.exists(datapath + '/x_test.npy'):
        x_test = np.load(datapath + '/x_test.npy')
        y_test = np.load(datapath + '/y_test.npy')
    else:
        files = {'test': testfile}
        dataset = {}
        for data_type in ['test']:
            print('make {} dataset'.format(data_type))
            x_list = []
            y_list = []
            datafile = open(files[data_type], 'r')
            datafile = datafile.readlines()
            for i in tqdm(range(1,len(datafile))):  # len(datafile))):
                line = datafile[i].split(",")
                x = cv2.imread(os.path.join(datapath, line[0]+'.jpg'))
                h, w, c = x.shape
                x = x[int(h / 6):, :]
                x = cv2.resize(x, (224, 224))
                x = x.astype('float32') / 255.0
                x = np.transpose(x, (2, 0, 1))  # Transpose the image array
                x_list.append(x)
                y_list.append(line[1])
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            num_cls = len(set(int(v) for v in y_list))
            y_list = np.eye(num_cls)[np.array(y_list, dtype=int)]
            np.save('{}/x_{}'.format(datapath, data_type), x_list)
            np.save('{}/y_{}'.format(datapath, data_type), y_list)
            dataset['x_{}'.format(data_type)] = x_list
            dataset['y_{}'.format(data_type)] = y_list
        x_test = dataset['x_test']
        y_test = dataset['y_test']
    mean_l2_train = 0
    mean_inf_train = 0
    for im in x_test:
        mean_l2_train += np.linalg.norm(im[:, :, 0].flatten(), ord=2)
        mean_inf_train += np.abs(im[:, :, 0].flatten()).max()
    mean_l2_train /= len(x_test)
    mean_inf_train /= len(x_test)
    min_, max_ = float(np.amin(x_test)), float(np.amax(x_test))
    return (x_test, y_test), min_, max_, mean_l2_train, mean_inf_train

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x
        


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba" or opt.dataset=="ISIC2019"or opt.dataset=="Echo":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        if not KORNIA_AVAILABLE:
            raise ImportError("kornia is required for PostTensorTransform. pip install kornia")
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)

class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field,
                transform=None, add_extension=None,
                 limit=None, random_subset_size=None,
                 split=None):
        """

        :param root: root of dataset
        :param csv_file: csv file of whole dataset
        :param image_field: 'image'
        :param target_field: 'label
        :param transform:
        :param add_extension: 'jpg'
        :param limit:
        :param random_subset_size: int,get random subset dataset
        :param split: TXT document stored the names of images
        """
        self.root = root

        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform

        self.add_extension = add_extension

        self.data = pd.read_csv(csv_file, sep=None)
        self.class_amount_dict = None

        # pdb.set_trace()
        # Split
        if split is not None:
            with open(split, 'r') as f:
                selected_images = f.read().splitlines()
            self.data = self.data[self.data[image_field].isin(selected_images)]
            self.data = self.data.reset_index()



        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes

        print('Found {} images from {} classes.'.format(len(self.data),
                                                        len(classes)))
        for class_name, idx in self.class_to_idx.items():
            n_images = dict(self.data[self.target_field].value_counts())# [class_idx: amount of class]
            self.class_amount_dict = n_images
            print("    Class '{}' ({}): {} images.".format(
                class_name, idx, n_images[class_name]))

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])
        if self.add_extension:
            path = path + self.add_extension

        sample = Image.open(path).convert('RGB')

        target = self.class_to_idx[self.data.loc[index, self.target_field]]
        if self.transform is not None:
            # print("transform exixts")
            sample_trans = self.transform(sample)
            # print('ok')


        return sample_trans, target

    def __len__(self):
        return len(self.data)

def get_isic():
    csv_path = str(config.get_data_path('ISIC2019_labels.csv'))
    root_path = str(config.get_data_path('ISIC_2019_Training_Input'))
    dataset = CSVDataset(root=root_path, csv_file=csv_path, image_field='image', target_field='label',
                       add_extension='.jpg',
                       split=str(config.get_data_path('train0.txt')))
    return dataset.data;
    
    

def get_dataloader(opt, train=True,set_ISIC2019='Train', pretensor_transform=False):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)
    elif opt.dataset=='ISIC2019':
        # csv_path = '/media/userdisk1/yf/ISIC2019/ISIC2019_grandtruethlabels.csv'
        # root_path = '/media/userdisk1/yf/ISIC2019/ISIC_2019_Training_Input/'
        csv_path = str(config.get_data_path('ISIC2019_grandtruethlabels.csv'))
        root_path = str(config.get_data_path('ISIC_2019_Training_Input'))
        if set_ISIC2019 == 'Train':
            dataset = CSVDataset(root=root_path, csv_file=csv_path, image_field='image', target_field='label',
                               transform=transform, add_extension='.jpg',
                               split=str(config.get_data_path('txt/train'+str(opt.split_idx)+'.txt')))
                                # split='/media/userdisk1/yf/ISIC2019/txt/train'+str(opt.split_idx)+'.txt')
        elif set_ISIC2019 == 'Val':
            dataset = CSVDataset(root=root_path, csv_file=csv_path, image_field='image', target_field='label',
                                 transform=transform, add_extension='.jpg',
                                 split=str(config.get_data_path('txt/validation'+str(opt.split_idx)+'.txt')))
        elif set_ISIC2019 == 'Test':
            dataset=CSVDataset(root=root_path, csv_file=csv_path, image_field='image', target_field='label',
                              transform=transform, add_extension='.jpg',
                              split=str(config.get_data_path('txt/test'+str(opt.split_idx)+'.txt')))
        else:
            print ('Wrong set_ISIC2019',set_ISIC2019)
    elif opt.dataset=='Echo':
        train_dir = str(config.get_data_path('Echocardiogram/data_split/train'))
        img_size = (128, 128)
        batch_size = 32
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        dataLoaded = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', color_mode='grayscale')
        dataset = ImageFolderDataset(dataLoaded)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=False)
    return dataloader

class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, directory_iterator):
        self.directory_iterator = directory_iterator

    def __len__(self):
        return len(self.directory_iterator)

    def __getitem__(self, index):
        x, y = self.directory_iterator[index]
        x = torch.stack([to_tensor(x[i]) for i in range(len(x))])
        return x, torch.tensor(y)

def get_dataset(opt, train=True):
    if opt.dataset == "gtsrb":
        dataset = GTSRB(
            opt,
            train,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(
            opt,
            split,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    else:
        raise Exception("Invalid dataset")
    return dataset


def main():
    pass


if __name__ == "__main__":
    main()
