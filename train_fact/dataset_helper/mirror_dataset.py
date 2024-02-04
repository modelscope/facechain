import torch
import json 
from PIL import Image
import random
import PIL
import numpy as np
import os 
import random
import torchvision.transforms.functional as TF
from torch.utils.data import RandomSampler, DataLoader, BatchSampler, Dataset
from torchvision.transforms import transforms
from torch.utils.data.distributed import  DistributedSampler
from typing import Optional
from torch import distributed as dist
import torch.multiprocessing as multiprocessing
import argparse
import math


from torch.utils.data import BatchSampler

class MyBatchSampler(BatchSampler):
    '''
    return same idx in each batch
    '''
    def _init__(self, sampler, batch_size, drop_last=True):
        super.__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        # self.shuffle = shuffle
        self.shuffle = None

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch = [idx for i in range(self.batch_size)]
            # print('hello, this is from rank {}, batch is {}'.format(dist.get_rank(), batch))
            yield batch
            batch = []
        

    def __len__(self):
        return len(self.sampler)

class MyDisBatchSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, batch_size: int, num_replicas: Optional[int] = None,
                rank: Optional[int] = None, shuffle: bool = True,
                seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        repeat_indices = []
        for i in indices:
            for j in range(self.batch_size):
                repeat_indices.append(i)
        assert len(repeat_indices) == self.num_samples * self.batch_size

        return iter(repeat_indices)
class MyDisMultiIDBatchSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, batch_size: int, id_size: int, num_replicas: Optional[int] = None,
                rank: Optional[int] = None, shuffle: bool = True,
                seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.id_size = id_size
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        repeat_indices = []
        for i in indices:
            for j in range(self.batch_size):
                repeat_indices.append(i)
        assert len(repeat_indices) == self.num_samples * self.batch_size

        return iter(repeat_indices)

    



class MIRRORDataset():
    def __init__(self, image_rootdir, face_rootdir, caption_rootdir, sample_num=3, prob_use_caption=1, image_size=512, face_size=112, augment=False):
        self.image_rootdir = image_rootdir
        self.face_rootdir = face_rootdir
        self.caption_rootdir = caption_rootdir
        self.sample_num = sample_num
        self.prob_use_caption = prob_use_caption
        self.image_size = image_size
        self.face_size = face_size
        self.augment = augment

        self.image_files = os.listdir(image_rootdir)
        self.face_files = os.listdir(face_rootdir)
        self.caption_files = os.listdir(caption_rootdir)

    def __getitem__(self, index):

        caption_file = self.caption_files[index]
        with open(os.path.join(self.caption_rootdir, caption_file)) as item:
            string = item.read()
            item = json.loads(string)

        num = len(item['imgs'])
        idxs = [i for i in range(num)]
        random.shuffle(idxs)
        if num < self.sample_num:
            while len(idxs) < self.sample_num:
                idxs.append(idxs[-1])
        else:
            idxs = idxs[:self.sample_num]
        
        for idx in idxs:

            img = Image.open(os.path.join(self.image_rootdir, item['imgs'][idx]).replace('tif', 'png')).convert('RGB')
            img = img.resize((self.image_size, self.image_size), PIL.Image.BICUBIC)
            img= (transforms.PILToTensor()(img).float()/255 - 0.5) / 0.5 # norm
            # imgs.append(transforms.PILToTensor()(img))

            face = Image.open(os.path.join(self.face_rootdir, item['imgs'][idx]).replace('tif', 'png')).convert('RGB')
            face = face.resize((self.face_size, self.face_size), PIL.Image.BICUBIC)
            face = (transforms.PILToTensor()(face).float()/255 - 0.5) / 0.5 # norm


            if random.uniform(0, 1) < self.prob_use_caption:
                caption = item['captions'][idx]
            else:
                caption = ''

            out = {'image': img,
                   'face': face,
                   'caption': caption,
                   'idx': idx,
                   'index': index}
            return out

    def __len__(self):
        return len(self.caption_files)



if __name__ == '__main__':

    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    # dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 4)
    # synchronize()
    image_rootdir = '/mnt/workspace/haoyu/data/mirror_dataset/sl_data/cropimg'
    face_rootdir = '/mnt/workspace/haoyu/data/mirror_dataset/sl_data/aligned_masked'
    caption_rootdir = '/mnt/workspace/haoyu/data/mirror_dataset/sl_data/caption'
    dataset = MIRRORDataset(image_rootdir=image_rootdir,
                            face_rootdir=face_rootdir,
                            caption_rootdir=caption_rootdir,
                            prob_use_caption=0.5)
    # sub_sampler = RandomSampler(dataset)
    sampler = MyDisBatchSampler(dataset, batch_size=3)
    # sampler = MyBatchSampler(sampler=sub_sampler, batch_size=2, drop_last=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True, sampler=sampler)
    dataset_iter = iter(loader)
    batch = dataset_iter.__next__()
    print(batch['image'].shape)
    print('hello, this is from rand {}, idx is {}, index is {}'.format(dist.get_rank(), batch['idx'], batch['index']) )
    # for i in range(dataset.__len__()):
    #     img_tensor = next(dataset_iter)['imgs'][0]
    #     img = img_tensor.cpu().clone().squeeze(0) # [3, 512, 512]
    #     img = img.mul(0.5).add(0.5) * 255 # renorm
    #     img = img.permute(1, 2, 0)
    #     print(img.max())
    #     print(img.shape)
    #     image = Image.fromarray(np.uint8(img.numpy()))
    #     image.save(save_dir + '{}.jpg'.format(i))
    #     print(save_dir + '{}.jpg'.format(i) + ' has been saved!')


