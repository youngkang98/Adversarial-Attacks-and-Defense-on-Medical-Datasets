import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv
import kornia.augmentation as A
import random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from torchvision.transforms.functional import to_tensor

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset


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
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba" or opt.dataset=="ISIC2019"or opt.dataset=="Echo" or opt.dataset == "COVID-19" or opt.dataset == "OCT" or opt.dataset == "CXRAY":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
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



def get_dataloader(opt, train=True,trainOrTestData='Train', pretensor_transform=False):
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
        csv_path = 'C:/Users/lkang/Documents/txt/ISIC2019_grandtruethlabels.csv'
        root_path = 'C:/Users/lkang/Documents/ISIC_2019_Training_Input/'
        if trainOrTestData == 'Train':
            dataset = CSVDataset(root=root_path, csv_file=csv_path, image_field='image', target_field='label',
                               transform=transform, add_extension='.jpg',
                               # split='C:/Users/lkang/Documents/txt/train'+str(opt.split_idx)+'.txt')
                                split='C:/Users/lkang/Documents/Adversarial-Attack-and-Defense/ISIC2019_train_012.txt')
                                # split='/media/userdisk1/yf/ISIC2019/txt/train'+str(opt.split_idx)+'.txt')
        elif trainOrTestData == 'Val':
            dataset = CSVDataset(root=root_path, csv_file=csv_path, image_field='image', target_field='label',
                                 transform=transform, add_extension='.jpg',
                                 split='C:/Users/lkang/Documents/txt/validation' + str(opt.split_idx) + '.txt')
        elif trainOrTestData == 'Test':
            dataset=CSVDataset(root=root_path, csv_file=csv_path, image_field='image', target_field='label',
                              transform=transform, add_extension='.jpg',
                              # split='C:/Users/lkang/Documents/txt/test' + str(opt.split_idx) + '.txt')
                              split='C:/Users/lkang/Documents/Adversarial-Attack-and-Defense/ISIC2019_test_012.txt')
        else:
            print ('Wrong set_ISIC2019',trainOrTestData)
    elif opt.dataset=='Echo':
        train_dir = "C:/Users/lkang/Documents/Echocardiogram/Echocardiogram/data_split/train"
        img_size = (128, 128)
        batch_size = 32
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        dataLoaded = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', color_mode='grayscale')
        dataset = ImageFolderDataset(dataLoaded)
    elif opt.dataset=='COVID-19':
        root_dir = 'C:/Users/lkang/Documents/COVID-Net/'+trainOrTestData
        test_file = 'test_COVIDx8A.txt'
        train_file = 'train_COVIDx8A.txt'
        if trainOrTestData == 'Train':
            dataset = COVID19Dataset(txt_file=train_file, root_dir=root_dir, transform=transform)
        elif trainOrTestData == 'Test':
            dataset=COVID19Dataset(txt_file=test_file, root_dir=root_dir, transform=transform)
        else:
            print ('Wrong set_ISIC2019',trainOrTestData)
    elif opt.dataset == 'OCT':
        train_data_path = 'C:/Users/lkang/Documents/OCT2017/train'
        test_data_path = 'C:/Users/lkang/Documents/OCT2017/test'
        testfile = 'C:/Users/lkang/Documents/Adversarial-Attack-and-Defense/OCT2017-test.csv'
        trainFile = 'C:/Users/lkang/Documents/Adversarial-Attack-and-Defense/OCT2017-train.csv'
        if trainOrTestData == 'Train':
            dataset = DatasetSeprateByClass(train_data_path, trainFile, 'train_data', transform=transform, one_hot_encode= False, num_classes=4)
        elif trainOrTestData == 'Test':
            dataset = DatasetSeprateByClass(test_data_path, testfile, 'test_data', transform=transform, one_hot_encode= False, num_classes=4)
    elif opt.dataset == 'CXRAY':
        train_data_path = 'C:/Users/lkang/Documents/chest_xray/train'
        test_data_path = 'C:/Users/lkang/Documents/chest_xray/test'
        testfile = 'C:/Users/lkang/Documents/Adversarial-Attack-and-Defense/CXRAY-test.csv'
        trainFile = 'C:/Users/lkang/Documents/Adversarial-Attack-and-Defense/CXRAY-train.csv'
        if trainOrTestData == 'Train':
            dataset = DatasetSeprateByClass(train_data_path, trainFile, 'train_data', transform=transform, one_hot_encode= False, num_classes=2)
        elif trainOrTestData == 'Test':
            dataset = DatasetSeprateByClass(test_data_path, testfile, 'test_data', transform=transform, one_hot_encode= False, num_classes=2)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
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
    
class COVID19Dataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.annotations = self.read_txt_file(txt_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    
    def read_txt_file(self, txt_file):
        annotations = []
        with open(txt_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 4:
                    _, image_name, label, _ = parts
                elif len(parts) == 3:
                    _, image_name, label = parts
                else:
                    continue
                annotations.append((image_name, label))
        return annotations

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name, label = self.annotations[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.label_mapping[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

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


def main():
    pass


if __name__ == "__main__":
    main()
