#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from datasets.CityScapesDataset import CityScapesDataset
from datasets.Gta5Dataset import Gta5Dataset
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm

import os
from utils import show_image_and_label
import sys


logger = logging.getLogger()


def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou

def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = torch.amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()

            # Specify the device type
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(19, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)

        # Upsampling layer to rescale the output to the size of the input
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2, inplace=True)
        x = self.classifier(x)

         # Upsample the final output
        x = self.upsample(x)


        return x
    
def train_uada(args, model, optimizer_g, dataloader_gta, dataloader_cityscapes, dataloader_val):
  output_file = 'training_output.log'
  with open(output_file, 'w') as f:
    # Save the original stdout and stderr so we can restore them later
    '''
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = f
    '''

    discriminator1 = torch.nn.DataParallel(Discriminator()).cuda()
    discriminator2 = Discriminator().cuda()
    discriminator3 = Discriminator().cuda()


    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()

    # Loss functions
    seg_loss_func = nn.CrossEntropyLoss(ignore_index=255)
    adv_loss_func = nn.BCEWithLogitsLoss()                        #loss uguale per entrambe le parti? cuh

    optimizer_d = torch.optim.Adam(discriminator1.parameters(), 0.001, betas = (0.9, 0.99))

    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr_g = poly_lr_scheduler(optimizer_g, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        lr_d = poly_lr_scheduler(optimizer_d, 0.001, iter=epoch, max_iter=args.num_epochs)

        model.train()
        discriminator1.train()
        discriminator2.train()
        discriminator3.train()

        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr_g %f, lr_d %f' % (epoch, lr_g, lr_d))
        loss_record_g = []
        loss_record_d = []
        for it, ((data_gta, label_gta), (data_cityscapes, _)) in enumerate(zip(dataloader_gta, dataloader_cityscapes)):
            data_gta = data_gta.cuda()
            label_gta = label_gta.long().cuda()
            data_cityscapes = data_cityscapes.cuda()

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            for param in discriminator1.parameters():
                param.requires_grad = False

            with amp.autocast():
                # Forward pass for segmentation network
                output_gta, out16_gta, out32_gta = model(data_gta)

                # Segmentation loss
                seg_loss1 = seg_loss_func(output_gta, label_gta.squeeze(1))
                seg_loss2 = seg_loss_func(out16_gta, label_gta.squeeze(1))
                seg_loss3 = seg_loss_func(out32_gta, label_gta.squeeze(1))
                seg_loss = seg_loss1 + seg_loss2 + seg_loss3

            scaler.scale(seg_loss).backward()
            scaler.step(optimizer_g)
            scaler.update()

            with amp.autocast():
                output_cityscapes, out16_cityscapes, out32_cityscapes = model(data_cityscapes)

            optimizer_g.zero_grad()


            with amp.autocast():

                # Forward pass for discriminator
                d_out_s1 = discriminator1(output_cityscapes)

                d_label_s = torch.ones_like(d_out_s1).cuda()

                # Discriminator loss
                loss_d_s1 = adv_loss_func(d_out_s1, d_label_s)

            weighted_loss = args.lambda_adv * loss_d_s1

            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer_g)
            scaler.update()


            for param in discriminator1.parameters():
                param.requires_grad = True

            output_cityscapes = output_cityscapes.detach()
            output_gta = output_gta.detach()


            with amp.autocast():

                # Forward pass for discriminator
                d_out_s1 = discriminator1(output_gta)

                d_label_s = torch.ones_like(d_out_s1).cuda()

                # Discriminator loss
                loss_d_s1 = adv_loss_func(d_out_s1, d_label_s)

            optimizer_d.zero_grad()
            optimizer_g.zero_grad()

            scaler.scale(loss_d_s1).backward()
            scaler.step(optimizer_d)
            scaler.update()

            with amp.autocast():

                d_out_t1 = discriminator1(output_cityscapes)

                d_label_t = torch.zeros_like(d_out_t1).cuda()

                loss_d_t1 = adv_loss_func(d_out_t1, d_label_t)


            optimizer_d.zero_grad()
            optimizer_g.zero_grad()


            scaler.scale(loss_d_t1).backward()
            scaler.step(optimizer_d)
            scaler.update()

            loss_G = seg_loss + weighted_loss

            loss_D = loss_d_t1 + loss_d_s1


            tq.update(args.batch_size)
            tq.set_postfix(loss_G='%.6f' % loss_G, loss_D='%.6f' % loss_D)
            step += 1
            writer.add_scalar('loss_step_G', loss_G, step)
            writer.add_scalar('loss_step_D', loss_D, step)

            loss_record_g.append(loss_G.item())
            loss_record_d.append(loss_D.item())

        tq.close()

        loss_train_mean = np.mean(loss_record_g)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

#to get args from command line
def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode',
                       dest='mode',
                       type=str,
                       default='train',
    )
    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='STDCNet813',
    )
    parse.add_argument('--pretrain_path',
                      dest='pretrain_path',
                      type=str,
                      default='',
    )
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
    )
    parse.add_argument('--num_epochs',
                       type=int, default=50,
                       help='Number of epochs to train for')
    parse.add_argument('--epoch_start_i',
                       type=int,
                       default=0,
                       help='Start counting epochs from this number')
    parse.add_argument('--checkpoint_step',
                       type=int,
                       default=10,
                       help='How often to save checkpoints (epochs)')
    parse.add_argument('--validation_step',
                       type=int,
                       default=5,
                       help='How often to perform validation (epochs)')
    parse.add_argument('--crop_height',
                       type=int,
                       default=512,
                       help='Height of cropped/resized input image to modelwork')
    parse.add_argument('--crop_width',
                       type=int,
                       default=1024,
                       help='Width of cropped/resized input image to modelwork')
    parse.add_argument('--batch_size',
                       type=int,
                       default=8,
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='learning rate used for train')
    parse.add_argument('--num_workers',
                       type=int,
                       default=0,
                       help='num of workers')
    parse.add_argument('--num_classes',
                       type=int,
                       default=19,
                       help='num of object classes (with void)')
    parse.add_argument('--cuda',
                       type=str,
                       default='0',
                       help='GPU ids used for training')
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training')
    parse.add_argument('--save_model_path',
                       type=str,
                       default="./output_models",
                       help='path to save model')
    parse.add_argument('--optimizer',
                       type=str,
                       default='sgd',
                       help='optimizer, support rmsprop, sgd, adam')
    parse.add_argument('--loss',
                       type=str,
                       default='crossentropy',
                       help='loss function')
    #ours
    parse.add_argument('--citySpaces_path',
                       type=str,
                       default='',
                       help='path of cityScapes Dataset')
    parse.add_argument('--gta5_path',
                       type=str,
                       default='',
                       help='path of gta5 Dataset')
    
    # Handle unknown arguments (ignore them)
    args, unknown = parse.parse_known_args()

    return parse.parse_args()

def default():

    ## dataset
    n_classes = args.num_classes

    mode = args.mode

    train_dataset = CityScapesDataset(mode)
    dataloader_train = DataLoader(train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)
    

    val_dataset = CityScapesDataset(mode='val')
    dataloader_val = DataLoader(val_dataset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=args.num_workers,
                       drop_last=False)

    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    ## train loop
    train(args, model, optimizer, dataloader_train, dataloader_val)
    # final test
    val(args, model, dataloader_val)

def train_cityscapes(args):
    # Dataset: Cityscapes
    n_classes = args.num_classes  # Numero di classi semantiche

    citySpaces_path = args.citySpaces_path
    pretrainedModel_path = args.pretrain_path

    # Dataset di addestramento
    train_dataset = CityScapesDataset(root_dir=citySpaces_path, mode='train', dimension=(256, 128))
    dataloader_train = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=False,
                                drop_last=True)
    print(args.batch_size)

    # Dataset di validazione
    val_dataset = CityScapesDataset(root_dir=citySpaces_path, mode='val', dimension=(256, 128))
    dataloader_val = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers,
                                drop_last=False)

    # Modello: STDC pre-addestrato su ImageNet
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=pretrainedModel_path)

    # Utilizzo della GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model = model.to(device)

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')

    ## train loop
    train(args, model, optimizer, dataloader_train, dataloader_val)
    # final test
    val(args, model, dataloader_val)

def train_gta(args):
    ## dataset
    n_classes = args.num_classes
    mode = args.mode
    
    train_dataset = Gta5Dataset(root = args.gta5_path)
    dataloader_train = DataLoader(train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)

    # Access the first image and label directly from the dataset
    image, label = train_dataset[0]
    # Display the image and label using the same function as before
    show_image_and_label(image, label)

    # Load a batch of images and labels from the DataLoader
    val_dataset = Gta5Dataset(root = args.gta5_path)
    dataloader_val = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.num_workers,
                        drop_last=False)

    print(len(train_dataset))
    print(len(val_dataset))

    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')

    ## train loop
    train(args, model, optimizer, dataloader_train, dataloader_val)
    # final test
    val(args, model, dataloader_val)

def train_gta_augmentation(args):
    ## dataset
    n_classes = args.num_classes
    mode = args.mode
    
    train_dataset = Gta5Dataset(root = args.gta5_path, augmentation="color_jitter")
    dataloader_train = DataLoader(train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)

    # Access the first image and label directly from the dataset
    image, label = train_dataset[0]
    # Display the image and label using the same function as before
    show_image_and_label(image, label)

    # Load a batch of images and labels from the DataLoader
    val_dataset = Gta5Dataset(root = args.gta5_path)
    dataloader_val = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.num_workers,
                        drop_last=False)

    print(len(train_dataset))
    print(len(val_dataset))

    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')

    ## train loop
    train(args, model, optimizer, dataloader_train, dataloader_val)
    # final test
    val(args, model, dataloader_val)

def test_gta_on_cityscapes_val(args):
    n_classes = args.num_classes
    citySpaces_path = args.citySpaces_path
    pretrainedModel_GTA_path = args.pretrain_path

    # Modello: STDC pre-addestrato su ImageNet con fine tuning su GTA5
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=pretrainedModel_GTA_path)
    model.eval()

    # Dataset di validazione
    val_dataset = CityScapesDataset(root_dir=citySpaces_path, mode='val', dimension=(256, 128))
    dataloader_val = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers,
                                drop_last=False)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # final test
    val(args, model, dataloader_val)

def train_unsupervised_adversarial_domain_adaptation():
    import sys
    from torch.utils.data import DataLoader, Subset

    args = parse_args()

    # Redirect stdout to a file
    output_file = 'output.log'
    with open(output_file, 'w') as f:
        # Save the original stdout so we can restore it later
        original_stdout = sys.stdout
        sys.stdout = f

        ## dataset
        n_classes = args.num_classes
        mode = args.mode

        train_dataset = Gta5Dataset(mode="train")
        train_subset = Subset(train_dataset, range(1500))

        dataloader_train = DataLoader(train_subset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)

        # Access the first image and label directly from the dataset
        image, label = train_subset[0]

        # Display the image and label using the same function as before
        show_image_and_label(image, label)

        train_dataset2 = CityScapesDataset(mode="train")
        train_subset2 = Subset(train_dataset2, range(1500))

        dataloader_train2 = DataLoader(train_subset2,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)

        # Access the first image and label directly from the dataset
        image, label = train_subset2[0]

        # Display the image and label using the same function as before
        show_image_and_label(image, label)

        val_dataset = CityScapesDataset(mode="val")
        dataloader_val = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    drop_last=False)

        print(len(dataloader_train))
        print(len(dataloader_train2))
        print(len(dataloader_val))

        ## model
        model = BiSeNet(backbone=args.backbone, n_classes=n_classes,
                        pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)

        if torch.cuda.is_available() and args.use_gpu:
            model = torch.nn.DataParallel(model).cuda()

        ## optimizer
        # build optimizer
        if args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=5e-4)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        else:  # rmsprop
            print('not supported optimizer \n')

        ## train loop
        train(args, model, optimizer, dataloader_train, dataloader_train2, dataloader_val)
        # final test
        val(args, model, dataloader_val)

        # Restore stdout to its original state
        sys.stdout = original_stdout

    print(f"Output has been logged to {output_file}")


def main():

    massimo_args = parse_args()
    massimo_args.num_classes=19
    massimo_args.batch_size = 8
    massimo_args.num_workers = 2
    massimo_args.pretrain_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'STDCNet813M_73.91')
    massimo_args.citySpaces_path = os.path.join(os.path.dirname(__file__), 'CityScapes_ds')
    massimo_args.gta5_path = os.path.join(os.path.dirname(__file__), 'GTA5_ds')
    
    #punto1_1(massimo_args)
    punto1_2(massimo_args)


if __name__ == "__main__":
    main()