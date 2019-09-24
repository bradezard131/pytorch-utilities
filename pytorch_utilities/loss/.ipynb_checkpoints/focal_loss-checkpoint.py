import torch.nn.functional as F
from torch.autograd import Variable


def focal_loss(input, target, gamma=2, weight=None, ignore_index=-100, reduction='mean'):
    if input.dim() > 2:
        input = input.view(input.size(0), input.size(1), -1)
        input = input.transpose(1,2)
        input = input.contiguous().view(-1,input.size(2))
    target = target.view(-1, 1)

    logpt = F.log_softmax(input)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    loss = -( (1-pt) ** gamma * logpt )

    if weight is not None:
        weight = weight.gather(1, target[:,0])
        loss *= weight

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss
