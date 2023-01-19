import pytest
from src.eval import ModifiedResNet18


class TestModifiedResNet18:
    def test_load(self):
        my_net = ModifiedResNet18("models/working.pth")
        assert my_net.model != None

    def test_prediction(self):
        my_net = ModifiedResNet18("models/working.pth")
        assert my_net.predict("example_images/green_forest.jpg") == "no fire"
        assert my_net.predict("example_images/grass_fire.jpg") == "fire"
