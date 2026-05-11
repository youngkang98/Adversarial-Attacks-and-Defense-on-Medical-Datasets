import torch
import torch.nn as nn
import os
import argparse
from utils import supervisor, tools
from utils.dataloader import get_dataloader, DatasetSeprateByClass
import config
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import UniversalPerturbation,TargetedUniversalPerturbation
from art.defences.trainer import AdversarialTrainerFBFPyTorch,AdversarialTrainerMadryPGD
from art.data_generators import PyTorchDataGenerator

def calculate_dataset_stats(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    
    for data in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    
    return mean, std

def load_wanet_components(args):
    """
    Load WaNet model and transform components with proper device placement
    """
    # Set device
    device = torch.device(f'cuda:{args.devices}' if torch.cuda.is_available() else 'cpu')
    
    # Get transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Set up model
    if args.dataset == 'OCT':
        if (args.model_path.endswith('.pth.tar')):
            num_classes = 4
            model = models.resnet50()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # Load checkpoint
            try:
                state_dict = torch.load(args.model_path, map_location=device)
                model_state = state_dict["netC"]
                
                # Remove DataParallel prefix if present
                # if list(model_state.keys())[0].startswith('module.'):
                #     model_state = {k[7:]: v for k, v in model_state.items()}
                
                # Load weights and move model to device
                model.load_state_dict(model_state)
                model = model.to(device)
                
                print(f"Model loaded successfully and moved to {device}")

                # Load and move grids to device
                identity_grid = state_dict["identity_grid"].to(device)
                noise_grid = state_dict["noise_grid"].to(device)

                
                # Save grids
                poison_set_dir = supervisor.get_poison_set_dir(args)
                os.makedirs(poison_set_dir, exist_ok=True)
                torch.save(identity_grid.cpu(), os.path.join(poison_set_dir, 'identity_grid'))
                torch.save(noise_grid.cpu(), os.path.join(poison_set_dir, 'noise_grid'))
                
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        elif(args.model_path.endswith('.pt')):
            model = supervisor.get_arch(args)
            model.load_state_dict(torch.load(args.model_path))
            model = model.cuda()
        else:
            model = supervisor.get_arch(args)
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            # model.load_state_dict(torch.load(args.model_path))
            model = model.cuda()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
        

    
    # Get poison transform
    poison_transform = supervisor.get_poison_transform(
        poison_type='WaNet',
        dataset_name='OCT',
        target_class=config.target_class['OCT'],
        trigger_transform=data_transform,
        is_normalized_input=True,
        alpha=args.alpha if args.test_alpha is None else args.test_alpha,
        trigger_name=args.trigger,
        args=args
    )
    
    return model, poison_transform, data_transform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=False,
                        default='OCT',
                        choices=['cifar10', 'gtsrb', 'OCT'])
    parser.add_argument('-poison_type', type=str, required=False,
                        default='WaNet')
    parser.add_argument('-poison_rate', type=float, required=False,
                        default=0.05)
    parser.add_argument('-cover_rate', type=float, required=False,
                        default=0.1)
    parser.add_argument('-alpha', type=float, required=False,
                        default=0.15)
    parser.add_argument('-test_alpha', type=float, required=False, 
                        default=0.2)
    parser.add_argument('-trigger', type=str, required=False, default=None)
    parser.add_argument('-model_path', required=False, default='poisoned_train_set/OCT/WaNet_0.050_cover=0.100_poison_seed=0/AdvTrain_OCT2017_epoch1_clean_retrain.pth',
                        help='Path to the WaNet checkpoint file (.pth.tar)')
    parser.add_argument('-cleanser', type=str, required=False, default=None)
    parser.add_argument('-defense', type=str, required=False, default=None)
    parser.add_argument('-no_normalize', default=False, action='store_true')
    parser.add_argument('-no_aug', default=False, action='store_true')
    parser.add_argument('-devices', type=str, default='0')
    parser.add_argument('-seed', type=int, required=False, default=2333)
    
    args = parser.parse_args()
    
    args.input_height = 256
    args.input_width = 256
    args.input_channel = 3
    args.bs = 32
    args.num_workers = 2
    
    train_epoch =1
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device(f'cuda:{args.devices}' if torch.cuda.is_available() else 'cpu')
    
    # Load model and transforms
    model, poison_transform, data_transform = load_wanet_components(args)
    print(f"Loaded model from {args.model_path}")
    
    # Set up test dataset
    # kwargs = {'num_workers': 4, 'pin_memory': True}
    # test_set_dir = os.path.join('clean_set', args.dataset, 'clean_split')
    # test_set_img_dir = os.path.join(test_set_dir, 'data')
    # test_set_label_path = os.path.join(test_set_dir, 'clean_labels')
    
    # test_set = tools.IMG_Dataset(
    #     data_dir=test_set_img_dir,
    #     label_path=test_set_label_path,
    #     transforms=data_transform
    # )
    
    # test_loader = DataLoader(
    #     test_set,
    #     batch_size=32,
    #     shuffle=False,
    #     worker_init_fn=tools.worker_init,
    #     **kwargs
    # )
    
    test_loader = get_dataloader(args, False,trainOrTestData='Test')
    advtrain_loader = get_dataloader(args, False,trainOrTestData='Train')
    
    # mean,std = calculate_dataset_stats(test_loader)
    # print(f"mean:{mean} std:{std}")
    # Run evaluation
    print("Starting evaluation...")
    # tools.test(
    #     model=model,
    #     test_loader=test_loader,
    #     poison_test=True,
    #     poison_transform=poison_transform,
    #     num_classes=4,
    #     source_classes=None,
    #     all_to_all=('all_to_all' in args.poison_type)
    # )
    # tools.test_with_debug(
    #     model=model,
    #     test_loader=test_loader,
    #     poison_test=True,
    #     poison_transform=poison_transform,
    #     num_classes=4
    # )
    
    tools.test_attack_success_rate_2(
    model=model,
    test_loader=test_loader,
    poison_transform=poison_transform,
    num_classes=4
    )
    
    tools.test_poison_behavior(
    model=model,test_loader=test_loader,
    poison_transform=poison_transform,
    num_classes=4
    )
    
    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=5e-4)
    
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=(3,224,224),
        optimizer=optimizer,
        nb_classes=4,
        device_type='gpu',
        preprocessing=None
    )
    
    adv_trainer = AdversarialTrainerMadryPGD(nb_epochs = 30,
                                             eps = 4/255,
                                             eps_step=1 / 255, 
                                             classifier = classifier,
                                             batch_size=32)

    # adv_trainer.fit(adversarial_x,adversarial_y,nb_epochs=100)
    # Custom fit loop
    for epoch in range(train_epoch):  # number of epochs
        print(f"Epoch {epoch+1}/{train_epoch}")
        for images, labels in advtrain_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Convert labels to numpy arrays if needed
            labels_np = labels.cpu().numpy()

            # Fit the adversarial trainer on the batch
            adv_trainer.fit(images.cpu().numpy(), labels_np, nb_epochs=1)

    image_list = []

    success_rate = 0

    save_path = 'advtrain_model'
    # Save the model and optimizer state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    os.makedirs(save_path, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_path, 'AdvTrain_OCT2017_epoch1_clean_retrain.pth'))

    model.eval()
    
    tools.test_attack_success_rate_2(
    model=model,
    test_loader=test_loader,
    poison_transform=poison_transform,
    num_classes=4
    )
    
    tools.test_poison_behavior(
    model=model,test_loader=test_loader,
    poison_transform=poison_transform,
    num_classes=4
    )

if __name__ == "__main__":
    main()