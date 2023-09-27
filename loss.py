from monai.losses import DiceLoss

def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value


loss_function = DiceLoss(sigmoid=True, squared_pred=True)