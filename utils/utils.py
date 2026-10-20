
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
import torch
import torch.nn as nn
from medpy import metric
from PIL import Image
from scipy.ndimage import zoom
from segmentation_mask_overlay import overlay_masks
from thop import clever_format, profile


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))


def one_hot_encoder(input_tensor, dataset, n_classes=None):
    tensor_list = []
    if dataset == 'MMWHS':
        dict = [0, 205, 420, 500, 550, 600, 820, 850]
        for i in dict:
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    else:
        for i in range(n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # print(inputs)
        assert inputs.size() == target.size(
        ), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.assd(pred, gt)
        precision = metric.binary.precision(pred, gt)
        recall = metric.binary.recall(pred, gt)
        return dice, hd95, jaccard, asd, precision, recall
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 1, 0, 0, 0
    else:
        return 0, 0, 0, 0, 0, 0


def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1
    else:
        return 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None, data_idx=None):
    image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()
    if class_names == None:
        mask_labels = np.arange(1, classes)
    else:
        mask_labels = class_names
    cmaps = mcolors.CSS4_COLORS

    my_colors = ['red', 'darkorange', 'yellow', 'forestgreen', 'blue', 'purple',
                 'magenta', 'cyan', 'deeppink', 'chocolate', 'olive', 'deepskyblue', 'darkviolet']
    cmap = {k: cmaps[k]
            for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}

    if len(image.shape) == 4:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :, :]
            x, y = slice.shape[1], slice.shape[2]
            # Patch sampling for images where x or y is greater than patch_size
            patch_size_val = patch_size[0] if isinstance(
                patch_size, (list, tuple)) else patch_size
            patch_size_val = int(patch_size_val)
            if x > patch_size_val or y > patch_size_val:
                img = torch.from_numpy(slice).float(
                ).cuda().unsqueeze(0)  # (1, C, H, W)
                overlap = patch_size_val // 2
                stride = overlap
                H, W = x, y
                patches_h = (H - patch_size_val) // stride + 2
                patches_w = (W - patch_size_val) // stride + 2
                output_canvas = torch.zeros(
                    (1, label.shape[1], H, W), device=img.device)
                p1_canvas = torch.zeros(
                    (1, 2, H, W), device=img.device)
                p2_canvas = torch.zeros(
                    (1, 2, H, W), device=img.device)
                weight_canvas = torch.zeros((1, 1, H, W), device=img.device)
                net.eval()
                with torch.no_grad():
                    for i in range(patches_h):
                        for j in range(patches_w):
                            yy = min(i * stride, H - patch_size_val)
                            xx = min(j * stride, W - patch_size_val)
                            patch = img[:, :, yy:yy+patch_size_val,
                                        xx:xx+patch_size_val]
                            # Model expects (B, C, H, W)
                            P = net(patch)
                            outputs = 0.0
                            for idx in range(len(P)):
                                outputs += P[idx]
                            p1_canvas[:, :, yy:yy+patch_size_val,
                                      xx:xx+patch_size_val] += P[0]
                            p2_canvas[:, :, yy:yy+patch_size_val,
                                      xx:xx+patch_size_val] += P[1]
                            out_patch = torch.softmax(outputs, dim=1)
                            out_patch = torch.argmax(out_patch, dim=1)

                            # Place the output patch (assume out_patch shape: (1, classes, patch_size, patch_size))
                            output_canvas[:, :, yy:yy+patch_size_val,
                                          xx:xx+patch_size_val] += out_patch
                            weight_canvas[:, :, yy:yy+patch_size_val,
                                          xx:xx+patch_size_val] += 1
                    # Average overlapping regions
                    output_canvas = output_canvas / weight_canvas
                    p1_canvas = p1_canvas / weight_canvas
                    p2_canvas = p2_canvas / weight_canvas
                    # If output is multi-channel (e.g., 2 channels), convert to 1 channel by argmax
                    if output_canvas.shape[1] > 1:
                        out = torch.argmax(output_canvas, dim=1).squeeze(
                            0).cpu().detach().numpy()
                    else:
                        out = output_canvas.squeeze().cpu().detach().numpy()
                    pred = out

                    # p1_canvas = torch.argmax(p1_canvas, dim=1)
                    p1_canvas = p1_canvas.cpu().detach().numpy()
                    # p2_canvas = torch.argmax(p2_canvas, dim=1)
                    p2_canvas = p2_canvas.cpu().detach().numpy()
                    plt.imsave('p1_0.png', p1_canvas[0, 0], cmap='gray')
                    plt.imsave('p2_0.png', p2_canvas[0, 0], cmap='gray')
                    plt.imsave('p1_1.png', p1_canvas[0, 1], cmap='gray')
                    plt.imsave('p2_1.png', p2_canvas[0, 1], cmap='gray')
            else:
                # ...existing code...
                input = torch.from_numpy(slice).float().cuda().unsqueeze(0)
                net.eval()
                with torch.no_grad():
                    P = net(input)
                    outputs = 0.0
                    for idx in range(len(P)):
                        outputs += P[idx]
                    out = torch.argmax(torch.softmax(
                        outputs, dim=1), dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()
                    if x != patch_size[0] or y != patch_size[1]:
                        pred = zoom(
                            out, (x / patch_size[0], y / patch_size[1]), order=0)
                    else:
                        pred = out

            lbl = label[ind, :, :]
            masks = []
            for i in range(1, classes):
                masks.append(lbl == i)
            preds_o = []
            for i in range(1, classes):
                preds_o.append(pred == i)
            prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))

    # Convert image, prediction, and label to (h, w, c) before saving
    image = np.transpose(image, (0, 2, 3, 1))
    prediction = np.transpose(prediction, (0, 2, 3, 1))
    label = np.transpose(label, (0, 2, 3, 1))

    img_itk = sitk.GetImageFromArray(
        image.astype(np.float32), isVector=True)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    img_itk.SetSpacing((1, 1, z_spacing))
    prd_itk.SetSpacing((1, 1, z_spacing))
    lab_itk.SetSpacing((1, 1, z_spacing))
    sitk.WriteImage(prd_itk, test_save_path + '/' +
                    case + f"{data_idx}_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + '/' +
                    case + f"{data_idx}_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path +
                    '/' + case + f"{data_idx}_gt.nii.gz")
    return metric_list


def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                # previous using 0
                slice = zoom(
                    slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():

                P = net(input)
                # print(len(P))

                outputs = 0.0
                for idx in range(len(P)):
                    outputs += P[idx]
                out = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(
                        out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():

            P = net(input)

            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list


def val_single_volume_1out(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                # previous using 0
                slice = zoom(
                    slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                p1 = net(input)
                outputs = p1
                out = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(
                        out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            p1 = net(input)
            outputs = p1
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list
