from __future__ import annotations
from PIL import Image, ImageOps
from torchvision import transforms
IMAGE_SIZE = 224
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
class PadSquare:
    """Pickle-safe torchvision transform for Windows DataLoader workers."""
    def __call__(self, image: Image.Image) -> Image.Image:
        return pad_square(open_rgb(image))
def open_rgb(image: str | Image.Image) -> Image.Image:
    im = Image.open(image) if isinstance(image, str) else image
    im = ImageOps.exif_transpose(im); im.load(); return im.convert("RGB")
def pad_square(im: Image.Image, color=(0, 0, 0)) -> Image.Image:
    side = max(im.size); canvas = Image.new("RGB", (side, side), color); canvas.paste(im, ((side-im.width)//2, (side-im.height)//2)); return canvas
def inference_transform(size: int = IMAGE_SIZE):
    return transforms.Compose([PadSquare(), transforms.Resize((size, size)), transforms.ToTensor(), NORMALIZE])
def training_transform(size: int = IMAGE_SIZE):
    return transforms.Compose([PadSquare(), transforms.Resize((size, size)), transforms.RandomRotation(20), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1), transforms.RandomApply([transforms.GaussianBlur(3)], p=.1), transforms.ToTensor(), NORMALIZE])
