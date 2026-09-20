import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
logger = logging.getLogger()

# define resnet building blocks
class ResidualBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1): 
        super(ResidualBlock, self).__init__() 
        
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), 
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel)) 
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: 
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, padding =0, bias=False), 
                                          nn.BatchNorm2d(outchannel)) 
            
    def forward(self, x): 
        out = self.left(x) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        return out

    
# define resnet
class ResNet(nn.Module):
    
    def __init__(self, num_classes = 10):
        super(ResNet, self).__init__()
        self.inchannel = 32
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1, bias = False), 
                                  nn.BatchNorm2d(32), 
                                  nn.ReLU())
        
        self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride = 2)
        self.layer2 = self.make_layer(ResidualBlock, 128, 3, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 3, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride = 2)
        self.layer5 = self.make_layer(ResidualBlock, 1024, 3, stride = 1)
        self.maxpool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(1024*8*8, num_classes)
        
    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        # B, 1, 512, 512
        x = self.conv1(x) # B, 32, 512, 512
        
        x = self.layer1(x) # B, 64, 256, 256
        x = self.layer2(x) # B, 128, 128, 128
        x = self.maxpool(x) # B, 128, 64, 64
        x = self.drop(x)
        
        x = self.layer3(x) # B, 256, 32, 32
        x = self.layer4(x) # B, 512, 16, 16
        x = self.layer5(x) # B, 1024, 16, 16
        x = self.maxpool(x) # B, 1024, 8, 8
        x = self.drop(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    

def load_classifier(model_path, num_classes):
    classifier = ResNet(num_classes)
    classifier.load_state_dict(torch.load(model_path, map_location="cpu"))
    return classifier



class Classifier():
    def __init__(self, config):
        self.config = config
        self.device = 'cuda:' + str(config['gpu_idx']) if config['use_gpu'] else 'cpu'
        self.model = ResNet(config['num_classes'])
        self.model.to(self.device)
        self.lr = config['lr']
        self.wd = config['wd']
        self.writer = SummaryWriter(log_dir=os.path.join(self.config['log_dir'], self.config['exp']))
        if config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        if config['warm_up']:
            self.warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=config['warm_up_steps'])
        if config['lr_decay']:
            if config['lr_scheduler'] == 'cosine':
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['max_epoch'] * 310 * config['batch_size'] // 32 - config['warm_up_steps'])
            else:
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_scheduler_step_size'], gamma=config['lr_scheduler_gamma'])
        self.loss = torch.nn.CrossEntropyLoss()


    def train_step(self, data_feed, epoch, step):
        mask, label = data_feed
        mask = mask.to(self.device)
        label = label.to(self.device) 
        mask = mask.float()
        predict = self.model(mask)
        loss = self.loss(predict, label)
        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        avg_loss = loss.item()/mask.shape[0]
        logger.info(f'Epoch: {epoch} Step: {step} loss: {avg_loss}')
        return avg_loss

    
    def train(self, train_loader, val_loader):
        n_iter = 0
        for epoch in range(self.config['max_epoch']):
            total_loss = 0
            for step, data_feed in enumerate(train_loader):
                total_loss += self.train_step(data_feed, epoch + 1, step + 1)
                n_iter += 1
                if self.config['warm_up'] and n_iter <= self.config['warm_up_steps']:
                    self.warm_up_scheduler.step()
                elif self.config['lr_decay']:
                    self.lr_scheduler.step()
            self.writer.add_scalar('train/loss', total_loss/(len(train_loader)), global_step=epoch+1)
            
            val_loss, val_acc = self.validate(val_loader)
            self.writer.add_scalar('val/loss', val_loss, global_step=epoch+1)
            self.writer.add_scalar('val/acc', val_acc, global_step=epoch+1)
            self.save_model(epoch + 1)
            
            

    def save_model(self, epoch):
        save_path = os.path.join(self.config['log_dir'], self.config['exp'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(), os.path.join(save_path, 'epoch_' + str(epoch) + '.pth'))

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))


    def validate(self, val_loader):
        # validation batch size is fixed to 1
        self.model.eval()
        total_loss = 0
        logger.info('Start validation...')
        acc = 0
        total = 0
        B = val_loader.batch_size
        with torch.no_grad(): 
            for data_feed in tqdm(val_loader):
                mask, label = data_feed
                mask = mask.to(self.device)
                label = label.to(self.device)
                mask = mask.float()

                predict = self.model(mask)
                loss = self.loss(predict, label)
                total_loss += loss.cpu().item()
                
                classes = torch.argmax(predict, dim=1)
                acc += (classes == label).sum()
                total += B
        
        val_loss = total_loss/total
        val_acc = acc/total
        logger.info(f'Validation loss: {val_loss}')
        logger.info(f'Validation acc: {val_acc}')
        
        return val_loss, val_acc

