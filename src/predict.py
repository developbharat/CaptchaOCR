import torch
from common import decode
from PIL import Image
import string
from common import Network
from torchvision.transforms.functional import to_tensor


class CaptchaPredictor:
    def __init__(self,
                 model_path: str,
                 classes: list[str] | str,
                 input_shape: tuple[int, int, int],
                 device: torch.device = torch.device('cpu')):
        self.classes = classes
        self.device = device
        self.input_shape = input_shape

        # load model
        self.model = Network(class_count=len(classes), input_shape=input_shape)
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def channels_to_mode(self, channels: int):
        if channels == 1:
            return "L"
        elif channels == 3:
            return "RGB"
        elif channels == 4:
            return "RGBA"
        else:
            raise Exception("Invalid no of channels found in provided picture")

    def process_picture(self, picture: Image) -> torch.Tensor:
        # convert picture to required format
        picture = picture.convert(self.channels_to_mode(self.input_shape[2]))
        picture = picture.resize((self.input_shape[0], self.input_shape[1]))
        return to_tensor(picture)

    def predict(self, picture: Image) -> str:
        picture = self.process_picture(picture)
        output = self.model(picture.unsqueeze(0).to(self.device))
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
        return decode(output_argmax[0], classes=self.classes)


if __name__ == '__main__':
    characters = '-' + string.digits + string.ascii_uppercase
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CaptchaPredictor(model_path="data/models/captcha.pt",
                             classes=characters,
                             input_shape=(192, 64, 3),
                             device=device)

    result = model.predict(picture=Image.open("data/sample.png"))
    print(f"Prediction: {result}")
