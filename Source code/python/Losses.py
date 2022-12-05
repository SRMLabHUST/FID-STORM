import torch
from torch.nn import MSELoss,L1Loss
import numpy as np

## L1L2loss

#  Define a matlab like gaussian 2D filter
def matlab_style_gauss2D(shape=(7,7),sigma=1):
    """
    2D gaussian filter - should give the same result as:
    MATLAB's fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h.astype(dtype=np.float)
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h*2.0
    h = h.astype('float32')
    return h

# Expand the filter dimensions
gfilter = matlab_style_gauss2D(shape = (7,7),sigma=1).reshape([1,1,7,7])

# Combined MSE + L1 loss
def L1L2loss(heatmap_true,spikes_pred):
    # generate the heatmap corresponding to the predicted spikes
    heatmap_pred = torch.nn.functional.conv2d(spikes_pred,torch.tensor(gfilter,dtype=torch.float32).cuda(),stride=1,padding=3)
    # heatmaps MSE
    losses = MSELoss(reduction='mean')
    loss_heatmaps = losses(heatmap_true, heatmap_pred)

    L1 = L1Loss()

    # l1 on the predicted spikes
    loss_spikes = L1(spikes_pred, torch.zeros_like(spikes_pred).cuda())
    return loss_heatmaps + 0.01*loss_spikes


def MSEloss(heatmap_true,heatmap_pred):
    # generate the heatmap corresponding to the predicted spikes
    # heatmaps MSE
    losses = MSELoss(reduction='mean')
    loss_heatmaps = losses(heatmap_true, heatmap_pred)

    return loss_heatmaps

