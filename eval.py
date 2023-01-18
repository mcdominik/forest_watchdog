import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image

from utils import my_transforms, my_device


class ModifiedResNet18(nn.Module):
    CLASSES = ['fire_images', 'non_fire_images']
    def __init__(self, path) -> None:
        super().__init__()
        self.model = None
        self._load(path=path)

    def _load(self, path):
        self.model = models.resnet18(pretrained=True)
        #freeze all params
        for params in self.model.parameters():
            params.requires_grad_ = False
        #changed final layer to be binary
        nr_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(nr_filters, 1)
        self.model = self.model.to(my_device)
        self.model.load_state_dict(torch.load(path, map_location=torch.device(my_device)))
        print(f'my_device is {my_device} ')

    def pre_image(self, image_path):
        img = Image.open(image_path)
        img_normalized = my_transforms(img).float()
        img_normalized = img_normalized.unsqueeze_(0)
        # input = Variable(image_tensor)
        img_normalized = img_normalized.to(my_device)
        # print(img_normalized.shape)
        with torch.no_grad():
            self.model.eval()
            if torch.sigmoid(self.model(img_normalized.float())) < 0.5:
                print("Prediction : fire")
                return 'fire'
            else:
                print("Prediction : no fire")
                return 'no fire'

my_net = ModifiedResNet18('models/working.pth')
my_net.pre_image('example_images/forest2.jpg')