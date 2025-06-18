from torchvision import transforms

def get_default_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
