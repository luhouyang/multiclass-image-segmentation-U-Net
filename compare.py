import os ,sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

PRED_PATH = 'C:/Users/User/Desktop/Python/deep_learning/multiclass-image-segmentation-U-Net/saved_images/color/frankfurt_000001_062793.png'
TRUE_MASK_PATH = "D:/storage/gtFine_trainvaltest/gtFine/val/frankfurt/frankfurt_000001_062793_gtFine_color.png"
GROUD_TRUTH_PATH = "D:/storage/gtFine_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000001_062793_leftImg8bit.png"

pred = Image.open(PRED_PATH)
true_mask = Image.open(TRUE_MASK_PATH)
ground_truth = Image.open(GROUD_TRUTH_PATH)

pred = np.array(pred)
true_mask = np.array(true_mask)
ground_truth = np.array(ground_truth)

plt.subplot(3, 1, 1)
plt.imshow(ground_truth)

plt.subplot(3, 1, 2)
plt.imshow(true_mask)

plt.subplot(3, 1, 3)
plt.imshow(pred)

plt.show()