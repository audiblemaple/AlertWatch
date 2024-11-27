import torchvision.transforms as transforms

def get_preprocessing_pipeline():
    """
    Defines the preprocessing pipeline for face images.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

