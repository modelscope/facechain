import argparse
import torch
from omegaconf import OmegaConf
import numpy as np
import random
from trainer_val_film import Trainer
from distributed import synchronize
import os 
import torch.multiprocessing as multiprocessing
from modelscope.hub.snapshot_download import snapshot_download


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--IMAGE_ROOT", type=str, default='/mnt/data/haoyu/project/data_clean_0808_20/cropimg')
    parser.add_argument("--FACE_ROOT", type=str, default='/mnt/data/haoyu/project/data_clean_0808_20/aligned_masked')
    parser.add_argument("--CAPTION_ROOT", type=str, default='/mnt/data/haoyu/project/data_clean_0808_20/caption')

    parser.add_argument("--VAL_IMAGE_ROOT", type=str, default='/mnt/data/haoyu/project/clean_data_lyf/cropimg')
    parser.add_argument("--VAL_FACE_ROOT", type=str, default='/mnt/data/haoyu/project/clean_data_lyf/aligned_masked')
    parser.add_argument("--VAL_CAPTION_ROOT", type=str, default='/mnt/data/haoyu/project/clean_data_lyf/caption')

    parser.add_argument("--DATA_ROOT", type=str,  default="./haoyufirst/pretrained_model/SD_model", help="path to DATA")
    parser.add_argument("--OUTPUT_ROOT", type=str,  default="OUTPUT", help="path to OUTPUT")

    parser.add_argument("--name", type=str,  default="test", help="experiment will be stored in OUTPUT_ROOT/name")
    parser.add_argument("--seed", type=int,  default=123, help="used in sampler")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--yaml_file", type=str,  default="configs/mirror.yaml", help="paths to base configs.")


    parser.add_argument("--base_learning_rate", type=float,  default=5e-5, help="")
    parser.add_argument("--weight_decay", type=float,  default=0.0, help="")
    parser.add_argument("--warmup_steps", type=int,  default=10000, help="")
    parser.add_argument("--scheduler_type", type=str,  default='constant', help="cosine or constant")
    parser.add_argument("--batch_size", type=int,  default=3, help="")
    parser.add_argument("--id_batch", type=int,  default=1, help="")
    parser.add_argument("--face_batch", type=int,  default=3, help="")
    parser.add_argument("--workers", type=int,  default=1, help="")
    parser.add_argument("--official_ckpt_name", type=str,  default="majicmixRealistic_v6_aligned.ckpt", help="SD ckpt name and it is expected in DATA_ROOT, thus DATA_ROOT/official_ckpt_name must exists")
    parser.add_argument("--ckpt", type=lambda x:x if type(x) == str and x.lower() != "none" else None,  default=None, 
        help=("If given, then it will start training from this ckpt"
              "It has higher prioty than official_ckpt_name, but lower than the ckpt found in autoresuming (see trainer.py) "
              "It must be given if inpaint_mode is true")
    )
    # parser.add_argument("--TRAINED_MODEL", type=str,  default="/mnt/data/haoyu/project/face0/OUTPUT/25_maj_wofc/tag01/checkpoint_00312500.pth", help="the model waiting for val")
    parser.add_argument("--FACT_MODEL", type=str,  default="./haoyufirst/pretrained_model/mirror_adapter_25_maj_atom.pth", help="the fact parameter has been trained")

    parser.add_argument("--face_prob", type=float,  default=0.1, help="classifer-free guidance for face condition")
    
    parser.add_argument('--inpaint_mode', default=False, type=lambda x:x.lower() == "true", help="Train a GLIGEN model in inpaitning setting")
    parser.add_argument('--randomize_fg_mask', default=False, type=lambda x:x.lower() == "true", help="Only used if inpaint_mode is true. If true, 0.5 chance that fg mask will not be a box but a random mask. See code for details")
    parser.add_argument('--random_add_bg_mask', default=False, type=lambda x:x.lower() == "true", help="Only used if inpaint_mode is true. If true, 0.5 chance add arbitrary mask for the whole image. See code for details")
    
    parser.add_argument('--enable_ema', default=False, type=lambda x:x.lower() == "true")
    parser.add_argument("--ema_rate", type=float,  default=0.9999, help="")
    parser.add_argument("--total_iters", type=int,  default=312500, help="")
    parser.add_argument("--save_every_iters", type=int,  default=625, help="")
    parser.add_argument("--disable_inference_in_training", type=lambda x:x.lower() == "true",  default=False, help="Do not do inference, thus it is faster to run first a few iters. It may be useful for debugging ")


    args = parser.parse_args()
    assert args.scheduler_type in ['cosine', 'constant']

    


    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()



    config = OmegaConf.load(args.yaml_file) 
    config.update( vars(args) )
    config.total_batch_size = config.batch_size * n_gpu
    if args.inpaint_mode:
        config.model.params.inpaint_mode = True
    config.model.params.face_prob = args.face_prob

    
    trainer = Trainer(config)
    synchronize()

    # Train
    trainer.start_training()














