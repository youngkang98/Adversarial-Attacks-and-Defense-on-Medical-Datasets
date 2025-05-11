import torch
import torch.nn as nn
import os
import argparse
from utils import supervisor, tools
import config
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F

def load_wanet_components(args):
    """
    Load WaNet model and transform components
    """
    # Get transforms
    _, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)
    
    # Set up model
    if args.dataset == 'OCT':
        num_classes = 4
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(args.devices)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
        
    # Load checkpoint
    state_dict = torch.load(args.model_path)
    
    # Extract model weights and grids
    model_state = state_dict["netC"]
    identity_grid = state_dict["identity_grid"]
    noise_grid = state_dict["noise_grid"]
    
    # Load model weights
    model.load_state_dict(model_state)
    model = nn.DataParallel(model)
    model = model.cuda()
    
    # Save grids to expected directory
    poison_set_dir = supervisor.get_poison_set_dir(args)
    os.makedirs(poison_set_dir, exist_ok=True)
    torch.save(identity_grid, os.path.join(poison_set_dir, 'identity_grid'))
    torch.save(noise_grid, os.path.join(poison_set_dir, 'noise_grid'))
    
    # # Get the expected path for the model
    # model_path = supervisor.get_model_dir(args)
    # os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model state dict
    # print(f"Saving converted model to {model_path}")
    # torch.save(model_state, os.path.join(poison_set_dir, 'OCT.pt'))
    
    # return model_path
    
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
    # Parse arguments
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
    parser.add_argument('-model_path', required=False, default='models/OCT_all2all_morph.pth.tar',
                        help='Path to the WaNet checkpoint file (.pth.tar)')
    parser.add_argument('-cleanser', type=str, required=False, default=None)
    parser.add_argument('-defense', type=str, required=False, default=None)
    parser.add_argument('-no_normalize', default=False, action='store_true')
    parser.add_argument('-no_aug', default=False, action='store_true')
    parser.add_argument('-devices', type=str, default='0')
    parser.add_argument('-seed', type=int, required=False, default=2333)
    
    args = parser.parse_args()
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    
    # Load model and transforms
    model, poison_transform, data_transform = load_wanet_components(args)
    print(f"Loaded model from {args.model_path}")
    
    # Set up test dataset
    if args.dataset == 'OCT':
        kwargs = {'num_workers': 4, 'pin_memory': True}
        test_set_dir = os.path.join('clean_set', args.dataset, 'clean_split')
        test_set_img_dir = os.path.join(test_set_dir, 'data')
        test_set_label_path = os.path.join(test_set_dir, 'clean_labels')
        
        test_set = tools.IMG_Dataset(
            data_dir=test_set_img_dir,
            label_path=test_set_label_path, 
            transforms=data_transform
        )
        
        test_loader = DataLoader(
            test_set,
            batch_size=32,  # You can adjust batch size
            shuffle=False,
            worker_init_fn=tools.worker_init,
            **kwargs
        )
        
        num_classes = 4  # OCT specific
        
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    # Run evaluation
    print("Starting evaluation...")
    tools.test(
        model=model,
        test_loader=test_loader,
        poison_test=False,
        poison_transform=poison_transform,
        num_classes=num_classes,
        source_classes=None,
        all_to_all=('all_to_all' in args.poison_type)
    )

if __name__ == "__main__":
    main()
