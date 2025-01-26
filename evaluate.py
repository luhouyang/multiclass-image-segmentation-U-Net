from tqdm import tqdm

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from model import UNET
from utils import save_as_images, get_cityscape_data
from cityscapesScripts.cityscapesscripts.helpers.labels import trainId2label as t2l

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'

ROOT_DIR_CITYSCAPES = 'D:/storage/gtFine_trainvaltest'
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1024

# MODEL_PATH = 'C:/Users/User/Desktop/Python/deep_learning/multiclass-image-segmentation-U-Net/output/model.pth'
MODEL_PATH = 'C:/Users/User/Desktop/Python/deep_learning/multiclass-image-segmentation-U-Net/output/model_highres.pth'

EVAL = True
PLOT_LOSS = False


def save_predictions(data, model):
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data)):

            X, y, s = batch
            X, y = X.to(DEVICE), y.to(DEVICE)
            predictions = model(X)

            predictions = torch.nn.functional.softmax(predictions, dim=1)
            pred_labels = torch.argmax(predictions, dim=1)
            pred_labels = pred_labels.float()

            # remapping labels
            pred_labels = pred_labels.to('cpu')
            pred_labels.apply_(lambda x: t2l[x].id)
            pred_labels = pred_labels.to(DEVICE)

            pred_labels = transforms.Resize((1024, 2048))(pred_labels)

            # filename and location to save
            s = str(s)
            pos = s.rfind('/', 0, len(s))
            name = s[pos + 1:-18]
            global location
            location = 'saved_images/multiclass_1'

            save_as_images(pred_labels, location, name, multiclass=True)


def evaluate(path):
    T = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH),
                          interpolation=Image.NEAREST)
    ])

    val_set = get_cityscape_data(
        root_dir=ROOT_DIR_CITYSCAPES,
        split='val',
        mode='fine',
        relabelled=True,
        transforms=T,
        shuffle=True,
        eval=True,
    )

    unet = UNET(in_channels=3, classes=19).to(DEVICE)
    checkpoint = torch.load(path)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    save_predictions(val_set, unet)


def plot_losses(path):
    checkpoint = torch.load(path)
    losses = checkpoint['loss_values']
    epoch = checkpoint['epoch']
    epoch_list = list(range(epoch))

    plt.plot(epoch_list, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss over {epoch+1} epochs")
    plt.show()


if __name__ == '__main__':
    if EVAL:
        evaluate(MODEL_PATH)
    if PLOT_LOSS:
        plot_losses(MODEL_PATH)
