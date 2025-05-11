import os
import torch
from torch import nn
import torch.nn.functional as F
import random
from torchvision.utils import save_image
from config import poison_seed

"""
WaNet (static poisoning). https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
"""

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, cover_rate, path, identity_grid, noise_grid, s=0.5, k=4, grid_rescale=1, target_class=0):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate 
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0

        # number of images
        self.num_img = len(dataset)
        
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.identity_grid = identity_grid
        self.noise_grid = noise_grid
    
    def generate_poisoned_training_set(self, batch_size=32):
        torch.manual_seed(poison_seed)
        random.seed(poison_seed)
    
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()
    
        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison + num_cover]
        cover_indices.sort()
    
        poison_id = []
        cover_id = []
        label_set = []
        img_batches = []  # Store batches of images
        
        pt, ct, cnt = 0, 0, 0
    
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.img_size) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
    
        ins = torch.rand(1, self.img_size, self.img_size, 2) * 2 - 1
        grid_temps2 = grid_temps + ins / self.img_size
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)
    
        batch_imgs = []
        for i in range(self.num_img):
            img, gt = self.dataset[i]
    
            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                img = F.grid_sample(img.unsqueeze(0), grid_temps2, align_corners=True)[0]
                ct += 1
    
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class
                img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
                pt += 1
    
            batch_imgs.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt += 1
    
            if len(batch_imgs) == batch_size or i == self.num_img - 1:
                img_batches.append(torch.cat(batch_imgs, dim=0))
                batch_imgs = []
    
        label_set = torch.LongTensor(label_set)
        print("Poison indices:", poison_id)
        print("Cover indices:", cover_id)
    
        return img_batches, poison_id, cover_id, label_set


    # def generate_poisoned_training_set(self):
    #     torch.manual_seed(poison_seed)
    #     random.seed(poison_seed)

    #     # random sampling
    #     id_set = list(range(0,self.num_img))
    #     random.shuffle(id_set)
    #     num_poison = int(self.num_img * self.poison_rate)
    #     poison_indices = id_set[:num_poison]
    #     poison_indices.sort() # increasing order

    #     num_cover = int(self.num_img * self.cover_rate)
    #     cover_indices = id_set[num_poison:num_poison+num_cover] # use **non-overlapping** images to cover
    #     cover_indices.sort()

        
    #     img_set = []
    #     label_set = []
    #     pt = 0
    #     ct = 0
    #     cnt = 0

    #     poison_id = []
    #     cover_id = []


    #     grid_temps = (self.identity_grid + self.s * self.noise_grid / self.img_size) * self.grid_rescale
    #     grid_temps = torch.clamp(grid_temps, -1, 1)

    #     ins = torch.rand(1, self.img_size, self.img_size, 2) * 2 - 1
    #     grid_temps2 = grid_temps + ins / self.img_size
    #     grid_temps2 = torch.clamp(grid_temps2, -1, 1)

    #     for i in range(self.num_img):
    #         img, gt = self.dataset[i]

    #         # noise image
    #         if ct < num_cover and cover_indices[ct] == i:
    #             cover_id.append(cnt)
    #             img = F.grid_sample(img.unsqueeze(0), grid_temps2, align_corners=True)[0]
    #             ct+=1

    #         # poisoned image
    #         if pt < num_poison and poison_indices[pt] == i:
    #             poison_id.append(cnt)
    #             gt = self.target_class # change the label to the target class
    #             img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
    #             pt+=1

    #         # img_file_name = '%d.png' % cnt
    #         # img_file_path = os.path.join(self.path, img_file_name)
    #         # save_image(img, img_file_path)
    #         # print('[Generate Poisoned Set] Save %s' % img_file_path)
            
    #         img_set.append(img.unsqueeze(0))
    #         label_set.append(gt)
    #         cnt+=1

    #     img_set = torch.cat(img_set, dim=0)
    #     label_set = torch.LongTensor(label_set)
    #     poison_indices = poison_id
    #     cover_indices = cover_id
    #     print("Poison indices:", poison_indices)
    #     print("Cover indices:", cover_indices)

    #     # demo
    #     img, gt = self.dataset[0]
    #     img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
    #     save_image(img, os.path.join(self.path, 'demo.png'))

    #     return img_set, poison_indices, cover_indices, label_set


class poison_transform():

    def __init__(self, img_size, normalizer, denormalizer, identity_grid, noise_grid, s=0.5, k=4, grid_rescale=1, target_class=0,attack_mode='all2all'):

        self.img_size = img_size
        self.normalizer = normalizer
        self.denormalizer = denormalizer
        self.target_class = target_class
        self.attack_mode = attack_mode

        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.identity_grid = identity_grid.cuda()
        self.noise_grid = noise_grid.cuda()
        self.num_classes = 4

    def transform(self, data, labels):
        grid_temps = (self.identity_grid.to(data.device) + self.s * self.noise_grid.to(data.device) / self.img_size) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
    
        
        data, labels = data.clone(), labels.clone()
        if(self.denormalizer != None):
            data = self.denormalizer(data)
        data = F.grid_sample(data, grid_temps.repeat(data.shape[0], 1, 1, 1), align_corners=True)
        if(self.normalizer != None):
            data = self.normalizer(data)
        # labels[:] = self.target_class
        if self.attack_mode == "all2one":
             labels[:] = self.target_class
        elif self.attack_mode == "all2all":
             labels[:] = torch.remainder(labels + 1, self.num_classes)

        # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # denormalizer = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # # normalizer = transforms.Compose([
        # #     transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        # # ])
        # # denormalizer = transforms.Compose([
        # #     transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
        # #                             (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
        # # ])
        # save_image(denormalizer(data)[0], 'b.png')

        return data, labels