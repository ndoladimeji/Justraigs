import gc
import time
import datetime
import pandas as pd
import numpy as np
import random
from skimage import io
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# For reproducibility
def seed_everything(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 1234
seed_everything(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)

# Dataset class
class Justraigs(Dataset):
    def __init__(self, dataframe, is_train=True):
        self.dataframe = dataframe
        self.is_train = is_train

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if is_train else transforms.Identity(),
            transforms.RandomVerticalFlip() if is_train else transforms.Identity(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_name = self.dataframe['Eye ID'][index]
        image_path = f'train_images/{img_name}.jpg'
        image = io.imread(image_path)
        image = self.transform(image)
        additional_features = self.dataframe.iloc[index, 4:14].values.astype(np.float32)
        return image, additional_features

# Define the model for the new task
class ResNet50Network(nn.Module):
    def __init__(self):
        super().__init__()

        original_model = resnet50(pretrained=False)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.additional_features_classifier = nn.Linear(2048, 10)

    def forward(self, image):
        image = self.features(image)
        image = image.view(image.size(0), -1)
        additional_features_prediction = self.additional_features_classifier(image)
        return additional_features_prediction

# Instantiate the new model and load pretrained weights
model = ResNet50Network().to(device)

# Data object and Loader
train_df = pd.read_csv('referable_glaucoma_new.csv')
dataset = Justraigs(train_df, is_train=True)
loader = DataLoader(dataset, batch_size=20, shuffle=True)

def hamming_loss(true_labels, predicted_labels):
    """
    Calculate the Hamming loss between true and predicted labels.
    """
    loss = torch.ne(true_labels, predicted_labels).float().mean()
    return loss.item()

# Prepare for training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.BCEWithLogitsLoss()

epochs = 1

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    hamming_losses = 0

    for images, additional_features in loader:
        images, additional_features = images.to(device), additional_features.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, additional_features)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted_labels = torch.sigmoid(outputs).round()
        hamming_losses += hamming_loss(additional_features, predicted_labels)

    avg_loss = running_loss / len(loader)
    avg_hamming_loss = hamming_losses / len(loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Average Hamming Loss: {avg_hamming_loss:.4f}')

# Save the model   
torch.save(model.state_dict(), './modeljust.pt')