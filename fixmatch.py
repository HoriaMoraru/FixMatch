import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

CONFIG = {"seed": 420,
          "epochs": 200,
          "img_size": 64,
          "num_classes": 30,
          "train_batch_size": 128,
          "val_batch_size": 128,
          "learning_rate": 0.001,
          "num_workers": 2,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          # StepLR Scheduler hyperparameters
          "step_size": 10,
          "gamma": 0.95
          }

torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed(CONFIG["seed"])

ROOT_DIR = '/kaggle/input/classification-on-unlabeled-and-mislabeled-images/'
TRAIN_LABELED_DIR = os.path.join(ROOT_DIR, 'train/train/labeled_images/')
TRAIN_UNLABELED_DIR = os.path.join(ROOT_DIR, 'train/train/unlabeled_images/')
TEST_DIR = os.path.join(ROOT_DIR, 'test/test/images/')
SAVE_PATH = "best_model.pth"

df = pd.read_csv(os.path.join(ROOT_DIR, 'train_annotations.csv'))
df.head()

class_names = df.class_name.unique()
class_to_index_mapping = {}
index_to_class_mapping = {}
for i in range(CONFIG['num_classes']):
    class_to_index_mapping[class_names[i]] = i
    index_to_class_mapping[i] = class_names[i]
class_to_index_mapping


class CustomDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform, dataset_type='train'):
        self.transform = transform
        df = pd.read_csv(csv_path)
        self.data_dir = data_dir

        # Split training set into train and validation
        train_data = df.sample(frac=0.8, random_state=CONFIG['seed'])
        if dataset_type == 'train':
            self.data = train_data
        elif dataset_type == 'val':
            self.data = df.drop(train_data.index)

        # For submission use full training set
        elif dataset_type == 'full-train':
            self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.data_dir, row['image_name'])
        img_label = class_to_index_mapping[row.class_name]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        output = {'img': img, 'label': img_label}

        return output

# Dataset for loading unlabeled set and test set
class UnlabeledDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.data_dir = data_dir
        self.img_names = [filename for filename in sorted(
            os.listdir(self.data_dir))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_names[idx])

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        
        output = {'img': img, 'img_name': self.img_names[idx]}

        return output

class TransformFixMatch(object):
    def __init__(self, weak, strong):
        self.weak = weak
        self.strong = strong

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)

        return weak, strong

weak_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

strong_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(CONFIG['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def prepare_loaders():
    train_set = CustomDataset(
        csv_path=os.path.join(ROOT_DIR, 'train_annotations.csv'),
        data_dir=TRAIN_LABELED_DIR,
        transform=weak_transforms,
        dataset_type='train'
    )

    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG['train_batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    unlabeled_train_set = UnlabeledDataset(
        data_dir=TRAIN_UNLABELED_DIR,
        transform=TransformFixMatch(weak_transforms, strong_transforms)
    )
    
    unlabeled_train_loader = DataLoader(
        unlabeled_train_set,
        batch_size=CONFIG['train_batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    val_set = CustomDataset(
        csv_path=os.path.join(ROOT_DIR, 'train_annotations.csv'),
        data_dir=TRAIN_LABELED_DIR,
        transform=test_transforms,
        dataset_type='val'
    )

    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG['val_batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, unlabeled_train_loader

from torchvision.models import densenet121

class DensenetModel(nn.Module):
    def __init__(self, num_classes):
        super(DensenetModel, self).__init__()
        self.backbone = densenet121(pretrained=False)
        in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(in_features=in_features, out_features=num_classes, bias=True))
    
    def forward(self, x):
        x = self.backbone(x)
        return x

model = DensenetModel(num_classes=CONFIG['num_classes'])
model.to(CONFIG['device'])

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=CONFIG['step_size'], gamma=CONFIG['gamma'])

def train_fixmatch_epoch(model, labeled_dataloader, unlabeled_dataloader, device, optimizer, epoch):
    criterion_labeled = nn.CrossEntropyLoss()
    criterion_unlabeled = nn.CrossEntropyLoss(reduction='none') # loss per example

    threshold = 0.90 # predictions smaller than 90% confidence are filtered.

    model.train()

    total_train_loss = 0.0
    dataset_size = 0

    bar = tqdm(enumerate(unlabeled_dataloader), total=len(unlabeled_dataloader), colour='cyan', file=sys.stdout)

    labeled_iterator = iter(labeled_dataloader)

    epoch_loss = 0

    for step, data in bar:
        unlabeled_images_weak = data['img'][0].to(device)
        unlabeled_images_strong = data['img'][1].to(device)
        
        try:
          labeled = next(labeled_iterator)
          labeled_images = labeled['img']
          labels = labeled['label']
        except StopIteration as e:
            labeled_iterator = iter(labeled_dataloader)
            labeled = next(labeled_iterator)
            labeled_images = labeled['img']
            labels = labeled['label']

        labeled_images = labeled_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred_labeled = model(labeled_images)

        # get pseudo-labels, don't propagate gradients
        with torch.no_grad():
          pred_weak = model(unlabeled_images_weak)

          # get confidence as a probability
          pred_weak_confidence = torch.nn.functional.softmax(pred_weak, dim = -1)
          max_values, max_indices = torch.max(pred_weak_confidence, dim = -1)

          # filter out unconfident predictions
          fixmatch_mask = (max_values > threshold).float()

        pred_strong = model(unlabeled_images_strong)

        loss_labeled = criterion_labeled(pred_labeled, labels)
        loss_consistency = criterion_unlabeled(pred_strong, max_indices) * fixmatch_mask
        loss_consistency = loss_consistency.mean()

        loss = loss_labeled + loss_consistency

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        bar.set_postfix(Epoch=epoch, LabeledLoss=loss_labeled.item(), ConsistencyLoss = loss_consistency.item(), FractionMasked=(1 - fixmatch_mask.float().mean()).item())

    return epoch_loss


def valid_epoch(model, dataloader, device, epoch):
    model.eval()

    total_val_loss = 0.0
    dataset_size = 0

    correct = 0

    bar = tqdm(enumerate(dataloader), total=len(
        dataloader), colour='cyan', file=sys.stdout)
    for step, data in bar:
        images = data['img'].to(device)
        labels = data['label'].to(device)

        batch_size = images.shape[0]

        pred = model(images)
        loss = criterion(pred, labels)

        _, predicted = torch.max(pred, 1)
        correct += (predicted == labels).sum().item()

        total_val_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = np.round(total_val_loss / dataset_size, 2)

        accuracy = np.round(100 * correct / dataset_size, 2)

        bar.set_postfix(Epoch=epoch, Valid_Acc=accuracy, Valid_Loss=epoch_loss)

    return accuracy, epoch_loss


def build_submission(model, dataloader, device, submission_file):
    model.eval()
    
    all_predictions = []
    all_image_names = []

    for data in dataloader:
        images = data['img'].to(device)
        img_names = data['img_name']
        pred = model(images)
        _, predicted = torch.max(pred, 1)
        
        predicted = predicted.cpu().numpy().tolist()
        all_predictions.extend(predicted)
        all_image_names.extend(img_names)
    
    all_predictions = [index_to_class_mapping[prediction] for prediction in all_predictions]
    data = list(zip(all_image_names, all_predictions))
    submission_df = pd.DataFrame(data=data, columns=['image_name', 'class_name'])
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission saved to {submission_file}")
        
def run_training_fixmatch(model, labeled_trainloader, unlabeled_trainloader, testloader, optimizer, num_epochs):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    top_accuracy = 0.0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        train_loss = train_fixmatch_epoch(model, labeled_trainloader, unlabeled_trainloader, CONFIG['device'], optimizer, epoch)
        with torch.no_grad():
            val_accuracy, val_loss = valid_epoch(model, testloader, CONFIG['device'], epoch)
            if val_accuracy > top_accuracy:
                print(f"Validation Accuracy Improved ({top_accuracy} ---> {val_accuracy})")
                top_accuracy = val_accuracy
        print()
        

labeled_trainloader, val_loader, unlabeled_trainloader = prepare_loaders()

run_training_fixmatch(model, labeled_trainloader, unlabeled_trainloader, val_loader, optimizer, CONFIG['epochs'])

print("Loading best model for submission")
model.load_state_dict(torch.load(SAVE_PATH))

test_set = UnlabeledDataset(TEST_DIR, test_transforms)

test_loader = DataLoader(
        test_set,
        batch_size = CONFIG['val_batch_size'], 
        shuffle = False,
        num_workers = CONFIG['num_workers'], 
        pin_memory = True
)

build_submission(model, test_loader, CONFIG['device'], 'submission.csv')