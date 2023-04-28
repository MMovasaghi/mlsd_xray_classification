import time
import copy
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import sys
import shutil
import torch
import torch.nn.functional as F 
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')


import wandb
# ToDo: login token
wandb.login(key="08555c78c1bcf41b7a775a888cb1d80a43dd5480")



def data_preprocessing(config):
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
        os.mkdir('dataset/Normal')
        os.mkdir('dataset/Pneumonia')
    else:
        shutil.rmtree("dataset")
        os.mkdir('dataset')
        os.mkdir('dataset/Normal')
        os.mkdir('dataset/Pneumonia')

    cmd = 'cp chest_xray/test/NORMAL/* dataset/Normal/'
    os.system(cmd)
    cmd = 'cp chest_xray/test/PNEUMONIA/* dataset/Pneumonia/'
    os.system(cmd)
    cmd = 'cp chest_xray/train/NORMAL/* dataset/Normal/'
    os.system(cmd)
    cmd = 'cp chest_xray/train/PNEUMONIA/* dataset/Pneumonia/'
    os.system(cmd)
    cmd = 'cp chest_xray/val/NORMAL/* dataset/Normal/'
    os.system(cmd)
    cmd = 'cp chest_xray/val/PNEUMONIA/* dataset/Pneumonia/'
    os.system(cmd)

    first = len(os.listdir('chest_xray/test/NORMAL')) + \
            len(os.listdir('chest_xray/train/NORMAL')) + \
            len(os.listdir('chest_xray/val/NORMAL'))
    secound = len(os.listdir('dataset/Normal'))
    if first == secound:
        print(f"[Log] Normal data transfer successfully [file number: {first}]")

    first = len(os.listdir('chest_xray/test/PNEUMONIA')) + \
            len(os.listdir('chest_xray/train/PNEUMONIA')) + \
            len(os.listdir('chest_xray/val/PNEUMONIA'))
    secound = len(os.listdir('dataset/Pneumonia'))
    if first == secound:
        print(f"[Log] Pneumonia data transfer successfully [file number: {first}]")


    normal = [f"dataset/Normal/{l}" for l in os.listdir('dataset/Normal')]
    pneumonia = [f"dataset/Pneumonia/{l}" for l in os.listdir('dataset/Pneumonia')]
    labels = ['Normal']*len(normal)
    labels.extend(['Pneumonia']*len(pneumonia))
    image_path = normal
    image_path.extend(pneumonia)
    df = {'image': image_path, 'label': labels}
    df = pd.DataFrame(df).sample(frac=1).reset_index(drop=True)


    df_ = {}
    df_['train'], df_rem = train_test_split(df, test_size=0.2, random_state=42, 
                                            stratify=df['label'])
    df_['val'], df_['test'] = train_test_split(df_rem, test_size=0.5, 
                                            random_state=42, stratify=df_rem['label'])

    if os.path.exists('split_dataset'):
        shutil.rmtree("split_dataset")
    os.mkdir('split_dataset')

    for phase in ['train', 'val', 'test']:
        os.mkdir(f'split_dataset/{phase}')
        os.mkdir(f'split_dataset/{phase}/Pneumonia')
        os.mkdir(f'split_dataset/{phase}/Normal')
        tmp = []
        for image, label in zip(df_[phase]['image'], df_[phase]['label']):
            shutil.copyfile(image, f'split_dataset/{phase}/{label}/{image.split("/")[-1]}')
            tmp.append(f'split_dataset/{phase}/{label}/{image.split("/")[-1]}')
        df_[phase]['image'] = tmp


    def data_transforms(phase = None):
        if phase == 'train':
            data_T = transforms.Compose([
                    transforms.Resize(size=config['image_dim']),
                    transforms.RandomRotation(degrees = (-20,+20)),
                    transforms.CenterCrop(size=224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        
        elif phase == 'val' or phase == 'test':
            data_T = transforms.Compose([
                    transforms.Resize(size=config['image_dim']),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        return data_T


    dataset = {}
    dataset_size = {}
    dataloaders = {}
    for phase in ['train', 'val', 'test']:
        dataset[phase] = datasets.ImageFolder(f'split_dataset/{phase}',
                                            transform=data_transforms(phase))
        dataset_size[phase] = dataset[phase].__len__()
        if phase == 'test':
            dataloaders[phase] = DataLoader(dataset[phase], batch_size=1, shuffle=True)
        else:
            dataloaders[phase] = DataLoader(dataset[phase], batch_size=config['batch_size'], shuffle=True)

    print(f"Class labels: {dataset['train'].class_to_idx}")

    images, labels = next(iter(dataloaders['train']))
    print(f"Input image Shape: {images.shape}")
    print(f"Label Shape: {labels.shape}")
    print(f"Batch Size: {config['batch_size']}")
    return dataloaders, dataset_size, df_


class FC_net(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=128, out_dim=2):
        super(FC_net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.layers(x)
        out = self.softmax(x)
        return out


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_model(model, criterion, optimizer, earlyStopper,
                dataloaders, dataset_size, 
                num_epochs=25, device='cpu'):
    since = time.time()
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        # print(f'Epoch {epoch+1}/{num_epochs}')
        # print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            
            tmp = {f'{phase}_loss': epoch_loss, 
                   f'{phase}_acc': epoch_acc, 
                   'epoch': epoch}
            # wandb logging
            wandb.log(tmp)

            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()
        if earlyStopper.early_stop(history['val_loss'][-1]):
            print("EarlyStopping")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def save_results(model, history, config, base_path="output"):
    suffix_file_name = f"lr{config['lr']}_bs{config['batch_size']}"
    dt = datetime.now()
    directory = f"{base_path}/{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}"

    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if not os.path.exists(directory):
        os.mkdir(directory)

    history_file_path = f"{directory}/history_{suffix_file_name}.pkl"
    with open(history_file_path, 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[Log] history has been saved in \"{history_file_path}\"')

    model_file_path = f"{directory}/model_{suffix_file_name}.pt"
    torch.save(model.state_dict(), model_file_path)
    print(f'[Log] model has been saved in \"{model_file_path}\"')


def main_train_model(config=None):

    ts = int(datetime.timestamp(datetime.now()))
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/hyper_parameter_testing'):
        os.mkdir('results/hyper_parameter_testing')
    os.mkdir(f'results/hyper_parameter_testing/hw2_hp_run_{ts}')
    os.mkdir(f'results/hyper_parameter_testing/hw2_hp_run_{ts}/model')
    with wandb.init(entity='moh2023', project='mlsd', name=f"hw2_hp_run_{ts}", config=config):
        config_wandb = wandb.config
        myconfig = {
            'batch_size': 640,
            'image_dim': (224, 224),
            'lr': config_wandb.lr,
            'epoch_num': 100,
            'weight_decay': config_wandb.weight_decay,
            'result_path': f"results/hyper_parameter_testing/hw2_hp_run_{ts}",
            'model_path': f"results/hyper_parameter_testing/hw2_hp_run_{ts}/model/"
        }
        print("="*10)
        print(f"run config:\n> lr: {myconfig['lr']}\n> wd: {myconfig['weight_decay']}")
        print("="*10)

        dataloaders, dataset_size, df_ = data_preprocessing(config=myconfig)

        model = models.resnet50(weights='IMAGENET1K_V2')
        model.fc = FC_net()

        i = 0
        for param in model.parameters():
            if i >= 159:
                param.requires_grad = True
            else:
                param.requires_grad = False
            i += 1

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Device: {device}")

        earlyStopper = EarlyStopper(patience=10, min_delta=1e-1)

        optimizer = optim.Adam(model.parameters(), lr=myconfig['lr'], 
                            weight_decay=myconfig['weight_decay'])

        class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(df_['train']['label']), 
                                            y=np.array(df_['train']['label']))
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model, history = train_model(model, criterion, optimizer, earlyStopper, 
                                    dataloaders, dataset_size, num_epochs=myconfig['epoch_num'], device=device)

        save_results(model, history, config=myconfig, base_path=f"{myconfig['model_path']}")
        cmd = f"cp -r {myconfig['model_path']} {config_wandb.output_path}"
        os.system(cmd)



def main(argv):
    sweep_config = {'method': 'random'}
    # sweep_config = {'method': 'grid'}
    # sweep_config = {'method': 'bayes'}

    metric = {'name': 'val_acc', 
            'goal': 'minimize'}

    sweep_config['metric'] = metric

    hyper_parameters = {
        'lr': {
            'distribution': 'log_uniform_values',
            'max': 1e-1, 
            'min': 1e-4
        },
        'weight_decay': {
            'distribution': 'log_uniform_values', 
            'max': 1e-1, 
            'min': 1e-8
        },
        'output_path': {'value': argv[1]}
    }
    sweep_config['parameters'] = hyper_parameters
    sweep_id = wandb.sweep(sweep_config, project='mlsd')
    wandb.agent(sweep_id, main_train_model, count=10)


if __name__ == '__main__':
    main(sys.argv)