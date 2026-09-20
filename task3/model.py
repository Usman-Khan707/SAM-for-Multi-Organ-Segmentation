import torch
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
# from loss import FocalLoss, SoftDiceLoss
from loss import FocalLoss, SoftDiceLoss
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.functional import threshold, normalize
from classifier import load_classifier
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
logger = logging.getLogger()


def dice(pred, true, true_sum=None):
    if true_sum is None:
        true_sum = true.sum()
    _dice = 2 * ((pred * true).int().sum()) / (pred.sum() + true_sum)
    return _dice


class SAM_classification():
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(log_dir='logs/combined')
        self.device = 'cuda:' + str(config['gpu_idx']) if config['use_gpu'] else 'cpu'
        self.model_checkpoint = os.path.join(config['model_root_path'], config['model_checkpoint'])
        self.model = sam_model_registry[config['model_type']](checkpoint=self.model_checkpoint)
        self.model.to(self.device)
        self.classifier = load_classifier(config['classifier_path'], config['num_classes'])
        self.classifier.to(self.device)
        self.sigmoid = torch.nn.Sigmoid()
        self.lr = config['lr']
        self.wd = config['wd']
        if config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.mask_decoder.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            self.optimizer = torch.optim.AdamW(self.model.mask_decoder.parameters(), lr=self.lr, weight_decay=self.wd)
        if config['warm_up']:
            self.warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=config['warm_up_steps'])
        if config['lr_decay']:
            if config['lr_scheduler'] == 'cosine':
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['max_epoch'] * 310 * config['batch_size'] // 32 - config['warm_up_steps'])
            else:
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_scheduler_step_size'], gamma=config['lr_scheduler_gamma'])
        self.focal_loss = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
        self.dice_loss = SoftDiceLoss(p=config['dice_p'], smooth=config['dice_smooth'])
        self.iou_loss = torch.nn.MSELoss()
        self.classifier_loss = torch.nn.CrossEntropyLoss()
        self.transform = ResizeLongestSide(config['img_input_size'][0])
        self.best = 0.
        if config['prompt_type'] == 'single_point':
            self.prompt_labels = torch.tensor([1]).to(self.device)
        if config['prompt_type'] == 'multipoints':
            num0 = config['prompt_num'] // 2
            num1 = config['prompt_num'] - num0
            self.prompt_labels = torch.tensor([1] * num1 + [0] * num0).to(self.device)
        # for params in self.model.mask_decoder.transformer.parameters():
        #     params.requires_grad = False

    def train_step(self, data_feed, epoch, step):
        image, label, prompt, organ = data_feed
        batch_size = image.shape[0]
        if self.config['embedded']:
            ori_size = (512, 512)
        else:
            ori_size = (image.shape[-2], image.shape[-1])
        image = image.to(self.device)
        label = label.to(self.device)
        if self.config['prompt_type'] != 'grid_points':
            prompt = prompt.to(self.device)
        organ = organ.to(self.device)
        self.model.train()
        self.classifier.train()
        with torch.no_grad():    
            if self.config['embedded']:
                image_embedding = image.squeeze(1)
            else:
                image = self.transform.apply_image_torch(image.unsqueeze(1).float())
                image = image.repeat(1, 3, 1, 1)
                label = label.unsqueeze(1)
                input_image = self.model.preprocess(image)
                image_embedding = self.model.image_encoder(input_image)

            points, boxes, masks = None, None, None
            prompt_type = self.config['prompt_type']
            if prompt_type == 'bbox':
                boxes = self.transform.apply_boxes_torch(prompt, ori_size)
            elif prompt_type == 'mask':
                masks = prompt
            elif prompt_type == 'grid_points':
                points, point_labels = prompt
                points = points.to(self.device)
                point_labels = point_labels.to(self.device)
                points = self.transform.apply_coords_torch(points, ori_size)
                points = points, point_labels
            else:
                points = self.transform.apply_coords_torch(prompt, ori_size)
                # repeat self.prompt_labels for batch_size times
                point_labels = self.prompt_labels.unsqueeze(0).repeat(batch_size, 1)
                points = points, point_labels
            
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=masks,
            )

        low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.config['multimask'],
            )
        if self.config['multimask']:
            best_idx = torch.argmax(iou_predictions, dim=1)
            # best_idx : [batch_size]
            # low_res_masks : [batch_size, num_masks, H, W] -> [batch_size, 1, H, W]
            # iou_predictions : [batch_size, num_masks] -> [batch_size, 1]
            low_res_masks = low_res_masks[torch.arange(batch_size), best_idx, :, :].unsqueeze(1)
            iou_predictions = iou_predictions[torch.arange(batch_size), best_idx].unsqueeze(1)

        upscaled_masks = self.model.postprocess_masks(low_res_masks, self.config['img_input_size'], ori_size).to(self.device)
        loss_mask = self.sigmoid(upscaled_masks)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device)
        
        predict = self.classifier(binary_mask)
        gt = organ - 1
        classifier_loss = self.classifier_loss(predict, gt)

        foc_loss = self.focal_loss(loss_mask, label.unsqueeze(1))
        dic_loss = self.dice_loss(loss_mask, label.unsqueeze(1))
        mask_loss = foc_loss + self.config['dice_weight'] * dic_loss
        # mask_loss = self.dice_loss(binary_mask, label)
        iou = torch.sum(binary_mask * label.repeat(1, binary_mask.shape[1], 1, 1), dim=[-1, -2]) / torch.sum(label.repeat(1, binary_mask.shape[1], 1, 1), dim=[-1, -2])        
        iou_loss = self.iou_loss(iou_predictions, iou) 
        loss = mask_loss + iou_loss * self.config['iou_weight'] + classifier_loss
        
        logger.info(f'Epoch: {epoch} Step: {step} lr: {self.optimizer.param_groups[0]["lr"]}')
        logger.info(f'loss: {loss} focal_loss: {foc_loss} dice_loss: {dic_loss} mask_loss: {mask_loss} iou_loss: {iou_loss}')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del image, label, prompt, organ, image_embedding, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, upscaled_masks, binary_mask, iou
        if not self.config['embedded']:
            del input_image
        torch.cuda.empty_cache()
        
        return loss, foc_loss, dic_loss, mask_loss, iou_loss, classifier_loss
    
    def train(self, train_loader, val_loader):
        n_iter = 0
        for epoch in range(self.config['max_epoch']):
            for step, data_feed in enumerate(train_loader):
                if step % 200 == 0 and step > 0:
                    avg_loss, avg_mask_loss, avg_iou_loss, acc = self.validate(val_loader)
                    self.writer.add_scalar('val/loss', avg_loss, global_step=(epoch+1)*len(train_loader)+step)
                    self.writer.add_scalar('val/mask_loss', avg_mask_loss, global_step=(epoch+1)*len(train_loader)+step)
                    self.writer.add_scalar('val/iou_loss', avg_iou_loss, global_step=(epoch+1)*len(train_loader)+step)
                    self.writer.add_scalar('val/acc', acc, global_step=(epoch+1)*len(train_loader)+step)
                loss, foc_loss, dic_loss, mask_loss, iou_loss, classifier_loss = self.train_step(data_feed, epoch + 1, step + 1)
                self.writer.add_scalar('train/loss', loss, global_step=(epoch+1)*len(train_loader)+step)
                self.writer.add_scalar('train/foc_loss', foc_loss, global_step=(epoch+1)*len(train_loader)+step)
                self.writer.add_scalar('train/dic_loss', dic_loss, global_step=(epoch+1)*len(train_loader)+step)
                self.writer.add_scalar('train/mask_loss', mask_loss, global_step=(epoch+1)*len(train_loader)+step)
                self.writer.add_scalar('train/iou_loss', iou_loss, global_step=(epoch+1)*len(train_loader)+step)
                self.writer.add_scalar('train/class_loss', classifier_loss, global_step=(epoch+1)*len(train_loader)+step)
                n_iter += 1
                if self.config['warm_up'] and n_iter <= self.config['warm_up_steps']:
                    self.warm_up_scheduler.step()
                elif self.config['lr_decay']:
                    self.lr_scheduler.step()
            if epoch > 0 and epoch % self.config['save_epoch'] == 0:
                self.validate(val_loader)
                self.save_model(epoch + 1)
        logger.info('Training finished!')
        logger.info(f'Best mDice score: {self.best}')


    def save_model(self, epoch):
        save_path = os.path.join(self.config['log_dir'], self.config['exp'], self.config['time_stamp'] + '-main')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch == -1:
            torch.save(self.model.state_dict(), os.path.join(save_path, 'best.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(save_path, 'epoch_' + str(epoch) + '.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(save_path, 'classifier_epoch_' + str(epoch) + '.pth'))
        
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))


    def validate(self, val_loader):
        # validation batch size is fixed to 1
        self.model.eval()
        self.classifier.eval()
        dice_list = [0.] * (self.config['num_classes'] + 1)
        dice_cnt = [0] * (self.config['num_classes'] + 1)
        loss_list = []
        mask_loss_list = []
        iou_loss_list = []
        logger.info('Start validation...')
        acc = 0
        total = 0
        B = val_loader.batch_size
        i = 0
        for data_feed in tqdm(val_loader):
            i += 1
            if self.config['subset_valid'] and i > 200:
                break
            image, label, prompt, organ = data_feed
            ori_size = (image.shape[-2], image.shape[-1]) if not self.config['embedded'] else (512, 512)
            image = image.to(self.device)
            label = label.to(self.device)
            if self.config['prompt_type'] == 'grid_points':
                pass
            else:
                prompt = prompt.to(self.device)
            organ = organ.to(self.device)
            with torch.no_grad():    
                if self.config['embedded']:
                    image_embedding = image.squeeze(1)
                else:
                    image = self.transform.apply_image_torch(image.unsqueeze(1).float())
                    image = image.repeat(1, 3, 1, 1)
                    label = label.unsqueeze(1)
                    input_image = self.model.preprocess(image)
                    image_embedding = self.model.image_encoder(input_image)

                points, boxes, masks = None, None, None
                prompt_type = self.config['prompt_type']
                if prompt_type == 'bbox':
                    boxes = self.transform.apply_boxes_torch(prompt, ori_size)
                elif prompt_type == 'mask':
                    masks = prompt
                elif prompt_type == 'grid_points':
                    points, point_labels = prompt
                    points = points.to(self.device)
                    point_labels = point_labels.to(self.device)
                    points = self.transform.apply_coords_torch(points, ori_size)
                    points = points, point_labels
                else:
                    points = self.transform.apply_coords_torch(prompt, ori_size)
                    point_labels = self.prompt_labels.unsqueeze(0)
                    points = points, point_labels

                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=boxes,
                    masks=masks,
                )

                low_res_masks, iou_predictions = self.model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=self.config['multimask'],
                    )
                if self.config['multimask']:
                    best_idx = torch.argmax(iou_predictions, dim=1)
                    low_res_masks = low_res_masks[:, best_idx, :, :]
                    iou_predictions = iou_predictions[:, best_idx]

                upscaled_masks = self.model.postprocess_masks(low_res_masks, self.config['img_input_size'], ori_size).to(self.device)
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device)
                mask_loss = self.focal_loss(binary_mask, label) + self.config['dice_weight'] * self.dice_loss(binary_mask, label)
                iou = torch.sum(binary_mask * label.repeat(1, binary_mask.shape[1], 1, 1), dim=[-1, -2]) / torch.sum(label.repeat(1, binary_mask.shape[1], 1, 1), dim=[-1, -2])        
                iou_loss = self.iou_loss(iou_predictions, iou) 
                loss = mask_loss + iou_loss * self.config['iou_weight']
                loss_list.append(loss.cpu().item())
                mask_loss_list.append(mask_loss.cpu().item())
                iou_loss_list.append(iou_loss.cpu().item())
                dice_score = dice(binary_mask, label)
                dice_list[organ] += dice_score.cpu().item()
                dice_cnt[organ] += 1
                
                predict = self.classifier(label.float())
                classes = torch.argmax(predict, dim=1) + 1
                acc += (classes == organ).sum()
                total += B
                
                del image, label, prompt, organ, image_embedding, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, upscaled_masks, binary_mask, mask_loss, iou, iou_loss, loss
                if not self.config['embedded']:
                    del input_image
                torch.cuda.empty_cache()
                
        avg_loss = np.mean(loss_list)
        avg_mask_loss = np.mean(mask_loss_list)
        avg_iou_loss = np.mean(iou_loss_list)
        logger.info(f'Validation loss: {avg_loss}')
        logger.info(f'Validation mask loss: {avg_mask_loss}')
        logger.info(f'Validation iou loss: {avg_iou_loss}')
        logger.info(f'Validation acc: {acc/total}')

        dice_list = np.array([dice_list[i] / dice_cnt[i] if dice_cnt[i] != 0 else 0 for i in range(1, self.config['num_classes'] + 1)])
        sum_dice = 0.
        n_organs = 0
        for i in range(self.config['num_classes']):
            logger.info(f'Organ {i+1}: Dice score {dice_list[i]}')
            sum_dice += dice_list[i]
            if dice_list[i] != 0:
                n_organs += 1
        # exclude unseen organs
        m_dice = sum_dice / n_organs
        logger.info(f'Validation mDice score: {m_dice}')
        if m_dice > self.best:
            self.best = m_dice
            self.save_model(-1)
        return avg_loss, avg_mask_loss, avg_iou_loss, acc/total
