import argparse

import numpy as np
from art.attacks.evasion import UniversalPerturbation,TargetedUniversalPerturbation
from art.defences.trainer import AdversarialTrainerFBFPyTorch,AdversarialTrainerMadryPGD

from uap_utils import (get_foolingrate_rate, get_preds, make_adv_img, set_up,
                       show_confusion_matrix)

parser = argparse.ArgumentParser(description='COVID-Net Evaluation')
parser.add_argument('--weightspath', default='COVIDNet-CXR-3',
                    type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta',
                    type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model',
                    type=str, help='Name of model ckpts')
parser.add_argument('--datapath', default='C:\\Users\\lkang\\Documents\\COVID-Net',
                    type=str, help='Path to data folder')
parser.add_argument('--trainfile', default='train_COVIDx8A.txt',
                    type=str, help='Path to testfile')
# parser.add_argument('--trainfile', default='train.txt',
#                     type=str, help='Path to testfile')
parser.add_argument('--testfile', default='test_COVIDx8A.txt',
                    type=str, help='Path to testfile')
# parser.add_argument('--testfile', default='test.txt',
#                     type=str, help='Path to testfile')
parser.add_argument('--norm', type=str, default='inf')
parser.add_argument('--eps', type=float, default=0.02)

args = parser.parse_args()
# (x_train, y_train), (x_test, y_test), (mean_l2_train,
#                                        mean_inf_train), norm, eps, classifier = set_up(args)

(x_test, y_test), (mean_l2_train, mean_inf_train), norm, eps, classifier = set_up(args)

acc_train = 0
acc_train_adv =0
fr_train_adv=0
# # Generate adversarial examples

adv_crafter = UniversalPerturbation(
    classifier,
    attacker='fgsm',
    delta=0.000001,
    attacker_params={'targeted': False, 'eps': 0.001},
    max_iter=15,
    eps=eps,
    norm=norm)

# _ = adv_crafter.generate(x_train)
_ = adv_crafter.generate(x_test)
noise = adv_crafter.noise[0, :]
noise = noise.astype(np.float32)
np.save('output/nontargeted_uap', noise)

# # Evaluate the ART classifier on adversarial examples

# acc_train, preds_train = get_preds(classifier, x_train, y_train)
acc_test, preds_test = get_preds(classifier, x_test, y_test)

# x_train_adv = x_train + noise
x_test_adv = x_test + noise

# acc_train_adv, preds_train_adv = get_preds(classifier, x_train_adv, y_train)
acc_test_adv, preds_test_adv = get_preds(classifier, x_test_adv, y_test)
# fr_train_adv = get_foolingrate_rate(preds_train_adv, y_train)
fr_test_adv = get_foolingrate_rate(preds_test_adv, y_test)

acc_train = 0
acc_train_adc =0
fr_train_adv=0

print('=== train test ===')
print('acc {:.3f} {:.3f}'.format(acc_train, acc_test))
print('acc_adv {:.3f} {:.3f}'.format(acc_train_adv, acc_test_adv))
print('fooling_rate {:.3f} {:.3f}'.format(fr_train_adv, fr_test_adv))

print('=== train ===')
# show_confusion_matrix(np.argmax(y_train, axis=1), preds_train_adv)
print('=== test ===')
show_confusion_matrix(np.argmax(y_test, axis=1), preds_test_adv)

# # imshow

make_adv_img(x_test[0], noise, x_test_adv[0], 'output/nontargeted_uap.png')
