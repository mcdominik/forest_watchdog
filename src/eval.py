import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import io
from src.utils import my_transforms, my_device


class ModifiedResNet18(nn.Module):
    """Modified ResNet18, input shape [3,224, 224]
    last layer changed to binary
    Args
    -------
    path: path to saved model

    Methods
    -------
    predict: returns prediction of custom image
    """

    CLASSES = ["fire_images", "non_fire_images"]

    def __init__(self, path) -> None:
        super().__init__()
        self.model = None
        self._load(path=path)

    def __repr__(self):
        return f"Modified ResNet18 with model: {self.model}"

    def _load(self, path: str) -> None:
        """Load desired weights and print
        current device (gpu or cpu)"""
        self.model = models.resnet18(pretrained=True)
        # freeze all params
        for params in self.model.parameters():
            params.requires_grad_ = False
        # changed final layer to be binary
        nr_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(nr_filters, 1)
        self.model = self.model.to(my_device)
        self.model.load_state_dict(
            torch.load(path, map_location=torch.device(my_device))
        )
        print(f"my_device is {my_device} ")

    def predict(self, image) -> str:
        """Predict from custom image, you can try provide path
        instead of image
        Args
        -------
        image: image object or path to image

        returns -> 'fire' or 'no fire'
        """
        if type(image) == str:
            img = Image.open(image)
        else:
            img = Image.open(io.BytesIO(image))
        img_normalized = my_transforms(img).float()
        img_normalized = img_normalized.unsqueeze_(0)
        img_normalized = img_normalized.to(my_device)
        with torch.no_grad():
            self.model.eval()
            if torch.sigmoid(self.model(img_normalized.float())) < 0.5:
                return "fire"
            else:
                return "no fire"


if __name__ == "__main__":
    my_net = ModifiedResNet18("models/working.pth")
    my_net.predict("example_images/forest2.jpg")
