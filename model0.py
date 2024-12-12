import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from skimage import io
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore")

# For reproducibility
def seed_everything(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Justraigs(Dataset):
    def __init__(self, dataframe, is_train=True):
        self.dataframe = dataframe
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=(0, 360)) if is_train else (lambda x: x),
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
        label = self.dataframe['target'][index]
        return image, label

class ModifiedResNet50(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(self.resnet50.children())[:-2])
        
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.fc = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = ModifiedResNet50().to(device)

# Data preparation
train_df = pd.read_csv('Rtrain_new.csv')
dataset = Justraigs(train_df, is_train=True)
loader = DataLoader(dataset, batch_size=5, shuffle=True)

learning_rate = 0.0005
epochs = 1

# Initiate the model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
class_weights = torch.tensor([1.0, 26.3])
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

# Training loop
# === EPOCHS ===
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    all_probs = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = torch.sigmoid(out).squeeze()
        probs = torch.round(pred)
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    # Calculate sensitivity at 95% specificity on test data          
    all_probs = np.array(all_probs)
    fpr, tpr, thresholds = roc_curve(train_df['target'].values, all_probs)
    desired_specificity = 0.95
    idx = np.argmin(np.abs(fpr - (1 - desired_specificity)))
    threshold_at_specificity = thresholds[idx]
    sensitivity_at_specificity = tpr[idx]

    print(f"Sensitivity at {desired_specificity*100:.2f}% specificity: {sensitivity_at_specificity:.4f}")
    print(f"Loss: {running_loss / len(loader):.4f}")

# Save the model
torch.save(model.state_dict(), './model.pt')