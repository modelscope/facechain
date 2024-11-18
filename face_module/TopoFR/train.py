import argparse
import logging
import os

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from losses import ArcFace
from lr_scheduler import PolyScheduler
from partial_fc import PartialFC, PartialFCAdamW
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

####
from persistent_homology import *
import random
from timm.data.random_erasing import RandomErasing
import math

from torchvision import transforms
from GUM import *

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        #backend="gloo",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)   #GPU

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    train_loader = get_dataloader(
        cfg.rec,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size, num_classes=cfg.num_classes).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    margin_loss = ArcFace()


    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    opt = torch.optim.SGD(
        params=[{"params": backbone.parameters()}],
        lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    from torch import nn
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(reduction="none")

    ColorJitter = transforms.ColorJitter(brightness=[0.1,0.5], hue=[-0.1,0.3],contrast=[0.3,0.6],saturation=[0.2,0.5])
    Normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    Grayscale = transforms.Grayscale(num_output_channels=3)
    GaussianBlur = transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1,5))

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1

            img_randomaug = img.clone()
            probability = 0.2
            batch_index = 0
            for index in range(img.size()[0]):
                if random.random() <= probability:
                    img_single = img[batch_index, :, :, :]
                    aug_random = np.random.randint(4)
                    if aug_random == 0:
                        img_original = ( (img_single*0.5) + 0.5 )
                        img_gauss = GaussianBlur(img_original)
                        img_aug = Normalize (img_gauss)
                        img_randomaug[batch_index, :, :, :] = img_aug
                    elif aug_random == 1:
                        img_original = ( (img_single*0.5) + 0.5 )
                        img_gray = Grayscale(img_original)
                        img_aug = Normalize (img_gray)
                        img_randomaug[batch_index, :, :, :] = img_aug
                    elif aug_random == 2:
                        img_original = ( (img_single*0.5) + 0.5 )
                        img_color = ColorJitter(img_original)
                        img_aug = Normalize (img_color)
                        img_randomaug[batch_index, :, :, :] = img_aug
                    elif aug_random == 3:
                        random_erase = RandomErasing(probability=1)  # perform RandomErase data augmentation
                        img_aug = random_erase(img_single)
                        img_randomaug[batch_index, :, :, :] = img_aug
                batch_index = batch_index + 1

            logits, bottleneck_embedding = backbone(img_randomaug)
            margin_logits = margin_loss(logits,local_labels)
            loss_cls_sample = criterion(margin_logits, local_labels)

            softmax_gum = torch.nn.Softmax(dim=1)(margin_logits)
            entropy = Entropy(softmax_gum)
            entropy = entropy.cpu().detach().numpy()
            entropy[2*np.arange(len(entropy)//2)] = -1 * entropy[2*np.arange(len(entropy)//2)]
            sample_weight, GUM_pi, GUM_sigma = gauss_unif(entropy.reshape(-1,1))
            sample_weight = torch.tensor(sample_weight).cuda()

            probability_gt = torch.ones_like(local_labels)
            for i_ in range(local_labels.size()[0]):
                j_ = local_labels[i_]
                probability_gt[i_] = softmax_gum[i_][j_]

            temp = 1
            w1 = torch.pow((2-sample_weight),temp)
            w2 = (1-probability_gt)
            loss_topo = compute_topological_loss(img, bottleneck_embedding, use_grad=True)
            loss_cls = (w1 * w2 * loss_cls_sample).mean()

            loss = loss_cls + 0.1*(loss_topo)
            ####

            if cfg.fp16:
                amp.scale(loss).backward(retain_graph=True)
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        from torch2onnx import convert_onnx
        convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", type=str, help="py config file", default='configs/ms1mv2_r50.py')
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
