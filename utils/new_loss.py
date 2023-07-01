"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = ['MixSoftmaxCrossEntropyOHEMLoss_1']



# bisenetv1
class OhemCELoss_1(nn.Module):
    def __init__(self, ignore_index=-1, reduction='none', thresh=0.9, min_kept=110000, *args,**kwargs):  # min_kept 可以改为  110000
        super(OhemCELoss_1, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = min_kept
        self.ignore_lb = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, logits, labels):
        # N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

# bisenetv1 v2
class OhemCELoss_2(nn.Module):

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss_2, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)



# class OhemCrossEntropy2d(nn.Module):
#     def __init__(self, ignore_index=-1, reduction='mean',thresh=0.9, min_kept=110000, use_weight=False, **kwargs):  # min_kept 可以改为  110000
#
#     # def __init__(self, ignore_index=-1, reduction='mean',thresh=0.9, min_kept=131072, use_weight=False, **kwargs):110000
#         super(OhemCrossEntropy2d, self).__init__()
#         self.ignore_index = ignore_index
#         self.thresh = float(thresh)
#         self.min_kept = int(min_kept)
#         if use_weight:
#             weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
#                                         1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
#                                         1.0865, 1.1529, 1.0507])
#             self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,weight=weight, ignore_index=ignore_index)
#         else:
#             self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,ignore_index=ignore_index)
#         print(use_weight,min_kept)
#     def forward(self, pred, target):
#         n, c, h, w = pred.size()
#         target = target.view(-1)
#         valid_mask = target.ne(self.ignore_index)
#         target = target * valid_mask.long()
#         num_valid = valid_mask.sum()
#
#         prob = F.softmax(pred, dim=1)
#         prob = prob.transpose(0, 1).reshape(c, -1)
#
#         if self.min_kept > num_valid:
#             print("Lables: {}".format(num_valid))
#         elif num_valid > 0:
#             prob = prob.masked_fill_(~valid_mask, 1)
#             mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
#             threshold = self.thresh
#             if self.min_kept > 0:
#                 index = mask_prob.argsort()
#                 threshold_index = index[min(len(index), self.min_kept) - 1]
#                 if mask_prob[threshold_index] > self.thresh:
#                     threshold = mask_prob[threshold_index]
#             kept_mask = mask_prob.le(threshold)
#             valid_mask = valid_mask * kept_mask
#             target = target * kept_mask.long()
#
#         target = target.masked_fill_(~valid_mask, self.ignore_index)
#         target = target.view(n, h, w)
#
#         return self.criterion(pred, target)


class MixSoftmaxCrossEntropyOHEMLoss_1(OhemCELoss_1):
    def __init__(self, aux=True, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss_1, self).__init__(ignore_index=ignore_index,**kwargs)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss_1, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss_1, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss_1, self).forward(*inputs)


