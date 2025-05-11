import json
import os
import shutil
from time import time

import config
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18
from networks.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from utils.dataloader import PostTensorTransform, get_dataloader,CSVDataset
from utils.utils import progress_bar

from torchvision import datasets, transforms, models
import time

# Start the timer
start_time = time.time()

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None
        
    netC = models.resnet50()
    netC.fc = nn.Linear(netC.fc.in_features, opt.num_classes)
    netC = netC.to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt,attack = False):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    avg_acc_cross = 0
    numOfImage = 0
    num_bd = 0
    num_cross = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        
        if attack:
            # Create backdoor data
            num_bd = int(bs * rate_bd)
            num_cross = int(num_bd * opt.cross_ratio)
            # if total_bd > opt.maxBD:
            #     num_bd = 0
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
    
            ins = torch.rand(num_cross, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)
    
            inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)
    
            inputs_cross = F.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)
    
            # if total_bd < opt.maxBD:
            #     total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
            #     # total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
            # else:
            #     total_inputs = torch.cat([inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
            #     # total_targets = targets
            total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
            total_inputs = transforms(total_inputs)
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        else:
            total_inputs = inputs
            total_targets = targets
        # start = time()
        total_preds = netC(total_inputs)
        # total_time += time() - start

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs - num_bd - num_cross
        total_bd += num_bd
        total_cross += num_cross
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd + num_cross) :], dim=1) == total_targets[(num_bd + num_cross) :]
        )
        if attack :
            total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
            if num_cross:
                total_cross_correct += torch.sum(
                    torch.argmax(total_preds[num_bd : (num_bd + num_cross)], dim=1)
                    == total_targets[num_bd : (num_bd + num_cross)]
                )
                avg_acc_cross = total_cross_correct * 100.0 / total_cross
                
            avg_acc_bd = total_bd_correct * 100.0 / total_bd

        avg_acc_clean = total_clean_correct * 100.0 / total_clean

        avg_loss_ce = total_loss_ce / total_sample

        if num_cross:
            progress_bar(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}| Total inputs: {:.4f} | Total bd: {:.4f}".format(
                    avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross, total_sample, total_bd
                ),
            )
        elif attack:
            progress_bar(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Total inputs: {:.4f} | Total bd: {:.4f}".format(avg_loss_ce, avg_acc_clean, avg_acc_bd,total_sample, total_bd),
            )
        else:
            progress_bar(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean),
            )
        
        if not os.path.exists(opt.temps):
            os.makedirs(opt.temps)
        
        # if attack:
            # for image in inputs_bd:
            #     imageName = "backdoor_image_" + str(numOfImage) + ".png"
            #     path = os.path.join(opt.temps, imageName)
            #     torchvision.utils.save_image(image, path, normalize=True)
            #     numOfImage += 1;
            # for i in range(num_bd):
            #     clean_image = inputs[i].cpu()
            #     noise = (inputs_bd[i] - inputs[i]).cpu()  # Calculate noise
            #     noisy_image = inputs_bd[i].cpu()
            
            #     # Concatenate images: left=clean_image, middle=noise, right=noisy_image
            #     concatenated_image = torch.cat([clean_image, noise, noisy_image], dim=2)
            
            #     # Save the concatenated image
            #     imageName = "comparison_image_" + str(numOfImage) + ".png"
            #     path = os.path.join(opt.temps, imageName)
            #     torchvision.utils.save_image(concatenated_image, path, normalize=True)
            #     numOfImage += 1
        # Save image for debugging
        # if not batch_idx % 50:
        #     if not os.path.exists(opt.temps):
        #         os.makedirs(opt.temps)
        #     path = os.path.join(opt.temps, imageName)
        #     torchvision.utils.save_image(inputs_bd, path, normalize=True)

        # Image for tensorboard
    #     if batch_idx == len(train_dl) - 2:
    #         num_bd = int(bs * rate_bd)
    #         residual = inputs_bd - inputs[:num_bd]
    #         batch_img = torch.cat([inputs[:num_bd], inputs_bd, total_inputs[:num_bd], residual], dim=2)
    #         batch_img = denormalizer(batch_img)
    #         batch_img = F.upsample(batch_img, scale_factor=(4, 4))
    #         grid = torchvision.utils.make_grid(batch_img, normalize=True)

    # # for tensorboard
    # if not epoch % 1:
    #     tf_writer.add_scalars(
    #         "Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd, "Cross": avg_acc_cross}, epoch
    #     )
    #     tf_writer.add_image("Images", grid, global_step=epoch)

    schedulerC.step()


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    noise_grid,
    identity_grid,
    best_clean_acc,
    best_bd_acc,
    best_cross_acc,
    tf_writer,
    epoch,
    opt,
    attack = False,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)
            
            if attack:
                
                # Evaluate Backdoor
                grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
                grid_temps = torch.clamp(grid_temps, -1, 1)
    
                ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
                grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
                grid_temps2 = torch.clamp(grid_temps2, -1, 1)
    
                inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
                if opt.attack_mode == "all2one":
                    targets_bd = torch.ones_like(targets) * opt.target_label
                if opt.attack_mode == "all2all":
                    targets_bd = torch.remainder(targets + 1, opt.num_classes)
                preds_bd = netC(inputs_bd)
                total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
                
                acc_bd = total_bd_correct * 100.0 / total_sample
                
            acc_clean = total_clean_correct * 100.0 / total_sample

            # Evaluate cross
            if attack and opt.cross_ratio:
                inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
                preds_cross = netC(inputs_cross)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

                acc_cross = total_cross_correct * 100.0 / total_sample

                info_string = (
                    "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | Cross: {:.4f}| Total: {:.2f}".format(
                        acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc,total_sample
                    )
                )
            elif attack:
                info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
            else:
                info_string = "Clean Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc
                )
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if attack :
        if not epoch % 1:
            tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)
    else:
        if not epoch % 1:
            tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean}, epoch)

    # Save checkpoint
    # if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc):
    print(" Saving...")
    best_clean_acc = acc_clean
    if attack:
        best_bd_acc = acc_bd
    if attack and opt.cross_ratio:
        best_cross_acc = acc_cross
    else:
        best_cross_acc = torch.tensor([0])
    state_dict = {
        "netC": netC.state_dict(),
        "schedulerC": schedulerC.state_dict(),
        "optimizerC": optimizerC.state_dict(),
        "best_clean_acc": best_clean_acc,
        "best_bd_acc": best_bd_acc,
        "best_cross_acc": best_cross_acc,
        "epoch_current": epoch,
        "identity_grid": identity_grid,
        "noise_grid": noise_grid,
    }
    torch.save(state_dict, opt.ckpt_path)
    resultPath = "results"+str(epoch)+".txt"
    if attack:
        with open(os.path.join(opt.ckpt_folder, resultPath), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item(),
                "cross_acc": best_cross_acc.item(),
                "epoch": str(epoch)
            }
            json.dump(results_dict, f, indent=2)
    else:
        with open(os.path.join(opt.ckpt_folder, resultPath), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "epoch": str(epoch)
            }
            json.dump(results_dict, f, indent=2)

    return best_clean_acc, best_bd_acc, best_cross_acc


def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    elif opt.dataset == 'ISIC2019':
        opt.num_classes = 8
    elif opt.dataset == 'Echo':
        opt.num_classes = 3
    elif opt.dataset == 'COVID-19':
        opt.num_classes = 3
    elif opt.dataset == 'OCT':
        opt.num_classes = 4
    elif opt.dataset == 'CXRAY':
        opt.num_classes = 2
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "ISIC2019":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    elif opt.dataset == 'Echo':
        opt.input_height = 128
        opt.input_width = 128
        opt.input_channel = 3
    elif opt.dataset == "COVID-19":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3    
    elif opt.dataset == "OCT":
        opt.input_height = 256
        opt.input_width = 256
        opt.input_channel = 3     
    elif opt.dataset == "CXRAY":
        opt.input_height = 256
        opt.input_width = 256
        opt.input_channel = 3   
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True,trainOrTestData='Train')
    test_dl = get_dataloader(opt, False,trainOrTestData='Test')
    
    attack = True

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])
            schedulerC.load_state_dict(state_dict["schedulerC"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            best_cross_acc = state_dict["best_cross_acc"]
            epoch_current = state_dict["epoch_current"]
            identity_grid = state_dict["identity_grid"]
            noise_grid = state_dict["noise_grid"]
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0
        best_cross_acc = 0.0
        epoch_current = 0

        # Prepare grid
        ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
            .to(opt.device)
        )
        array1d = torch.linspace(-1, 1, steps=opt.input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)

        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt,attack)
        best_clean_acc, best_bd_acc, best_cross_acc = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl,
            noise_grid,
            identity_grid,
            best_clean_acc,
            best_bd_acc,
            best_cross_acc,
            tf_writer,
            epoch,
            opt,
            attack
        )


if __name__ == "__main__":
    main()
    #====================================================================
    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Time taken to run the code: {elapsed_time} seconds")

