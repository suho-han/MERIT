import argparse
import logging
import os
import random
import sys
import time

import autorootcwd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from lib.networks import (MaxViT, MaxViT4Out, MaxViT_CASCADE, MERIT_Cascaded,
                          MERIT_Parallel)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.drive_dataset import DRIVEDataset
from src.utils import load_dataloader
from utils.utils import test_single_volume

parser = argparse.ArgumentParser()

parser.add_argument('--volume_path', type=str,
                    default='data/DRIVE', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='DRIVE', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_DRIVE', help='list dir')

parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=5000,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256,
                    help='input patch size of network input')
parser.add_argument('--image_size', type=int, default=512,
                    help='input patch size of network input')

parser.add_argument('--is_savenii', action="store_true",
                    help='whether to save results during inference')

parser.add_argument('--test_save_dir', type=str,
                    default='predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=2222, help='random seed')

args = parser.parse_args()

# if (args.num_classes == 14):
#     classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta',
#                'inferior vena cava', 'portal vein and splenic vein', 'pancreas', 'right adrenal gland', 'left adrenal gland']
# else:
#     classes = ['spleen', 'right kidney', 'left kidney',
#                'gallbladder', 'pancreas', 'liver', 'stomach', 'aorta']
classes = [0, 1]


def inference(args, model, test_save_path=None):
    testloader = load_dataloader(args, mode="test", resize=False)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    csv_path = os.path.join(test_save_path, 'test_results.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path, exist_ok=True)

    metric_list = 0.0
    with open(csv_path, 'w') as f:
        f.write("case_name,dice,hd95,jacard,asd,precision,recall\n")
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name.split('/')[0], z_spacing=1, data_idx=i_batch)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f mean_precision %f mean_recall %f' % (i_batch, case_name, np.mean(
                metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3], np.mean(metric_i, axis=0)[4], np.mean(metric_i, axis=0)[5]))
            f.write(
                f"{case_name},{np.mean(metric_i, axis=0)[0]},{np.mean(metric_i, axis=0)[1]},{np.mean(metric_i, axis=0)[2]},{np.mean(metric_i, axis=0)[3]},{np.mean(metric_i, axis=0)[4]},{np.mean(metric_i, axis=0)[5]}\n")
    metric_list = metric_list / len(testloader)
    # for i in range(1, args.num_classes):
    #     logging.info(
    #         f"Mean class ({i}) {classes[i-1]} mean_dice {metric_list[i-1][0]:.6f} ,mean_jacard {metric_list[i-1][2]:.6f}, mean_precision {metric_list[i-1][4]:.6f},mean_recall {metric_list[i-1][5]:.6f}, mean_hd95 {metric_list[i-1][1]:.6f},  mean_asd {metric_list[i-1][3]:.6f}"
    #     )
    mean_f1 = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jacard = np.mean(metric_list, axis=0)[2]
    mean_asd = np.mean(metric_list, axis=0)[3]
    mean_precision = np.mean(metric_list, axis=0)[4]
    mean_recall = np.mean(metric_list, axis=0)[5]
    with open(csv_path, 'a') as f:
        f.write(
            f"mean,{mean_f1},{mean_hd95},{mean_jacard},{mean_asd},{mean_precision},{mean_recall}\n")
    logging.info('Testing performance in best val model: mean_dice : %.6f mean_hd95 : %.6f, mean_jacard : %.6f mean_asd : %.6f mean_precision: %.6f mean_recall: %.6f' % (
        mean_f1, mean_hd95, mean_jacard, mean_asd, mean_precision, mean_recall))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'DRIVE': {
            'root_path': args.volume_path,
            'num_classes': args.num_classes,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'MERIT_Cascaded_Small_loss_MUTATION_w3_7_' + \
        dataset_name + str(args.img_size)
    snapshot_path = "MERIT/model_pth/{}/{}".format(
        args.exp, 'MERIT_Cascaded_Small_loss_MUTATION_w3_7')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_' + \
        str(args.max_iterations)[
            0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + \
        str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+'16'
    snapshot_path = snapshot_path + '_lr' + \
        str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s' + \
        str(args.seed) if args.seed != 1234 else snapshot_path

    # current_time = time.strftime("%H%M%S")
    current_time = '184642'
    print("The current time is", current_time)
    # replace 'current_time' with the 'current_time' in your trained folder from 'model_pth' directory
    snapshot_path = snapshot_path + '_run'+current_time

    net = MERIT_Cascaded(n_class=args.num_classes, img_size_s1=(args.img_size, args.img_size), img_size_s2=(
        256, 256), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()

    snapshot = os.path.join(snapshot_path, 'best.pth')
    snapshot = snapshot.replace('best', 'epoch_'+str(args.max_epochs-1))
    logging.info("Loading model from {}".format(snapshot))
    # if not os.path.exists(snapshot):
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'MERIT/test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # if args.is_savenii:
    args.test_save_dir = os.path.join(snapshot_path, "predictions")
    test_save_path = args.test_save_dir
    os.makedirs(test_save_path, exist_ok=True)
    # else:
    #     test_save_path = None
    inference(args, net, test_save_path)
