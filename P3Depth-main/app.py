import argparse
import os
from datetime import datetime
import sys
import random
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf, DictConfig, open_dict
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from src.data import define_dataset
from src.litmodel import DepthLitModel
from src.utils import load_config, print_config,update_config, check_machine, create_eval_dirs
from src.callback import DepthPredictionLogger
from torch.utils.data import Dataset, DataLoader
WANDB_PJ_NAME = 'p3depth'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a predictor')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config path. `configs/default.yaml` is loaded by default.')
    parser.add_argument('--model_config', type=str, default=None)
    parser.add_argument('--dataset_config', type=str, default=None)
    parser.add_argument('--exp_config', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='the checkpoint file to resume from')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu-ids', type=int, default=None, nargs='+')
    group.add_argument('--n_gpu', type=int, default=None)
    parser.add_argument("--amp", default=None, help="amp opt level", choices=['O1', 'O2', 'O3'])
    parser.add_argument("--profiler", default=None, help="'simple' or 'advanced'", choices=['simple', 'advanced'])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true")
    # parser.add_argument("--test_path",  type=str, default=None, help='test checkpoint path.')
    parser.add_argument("--data_dir", type=str, default=None, help='data path euler.')
    parser.add_argument("--out_dir", type=str, default=None, help='output path euler.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Overwrite configs. (ex. OUTPUT_DIR=results, SOLVER.NUM_WORKERS=8)')
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    def get_gpus(args: argparse.Namespace):
        if args.gpu_ids is not None:
            gpus = args.gpu_ids
        elif args.n_gpu is not None:
            gpus = args.n_gpu
        else:
            gpus = 1
        gpus = gpus if torch.cuda.is_available() else None
        return gpus

    # config
    config: DictConfig = load_config(args.config, args.model_config, args.dataset_config, args.exp_config, update_dotlist=args.opts)
    # Change paths based on machine
    config: DictConfig =  check_machine( config, args.data_dir, args.out_dir)
    ## REPRODUCIBILITY
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    # torch.use_deterministic_algorithms(True)



    

    #ToTensor
    def _is_pil_image(img):
        return isinstance(img, Image.Image)

    def _is_numpy_image(img):
        return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
    def to_tensor( pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

    
    
    class MyDataset(Dataset):
        def __init__(self, record_path):
        ## record_path:记录图片路径及对应label的文件
            self.data = []
            with open(record_path) as fp:
                for line in fp.readlines():
                    if line == '\n':
                        break
                    else:
                        tmp = line
                    ## tmp图片的路径
                        self.data.append(tmp)
        # 定义transform，将数据封装为Tensor
        #self.transform = transform

        # 获取单条数据
        def __getitem__(self, index):
            image_path='/dataset/kitti/data/'+self.data[index]
            image_path=image_path.replace('\n','')
            image1 = Image.open(image_path)
            image=image1.crop((10,628,1224,980))
            #image = image.resize((1216,352))
            image = np.asarray(image, dtype=np.float32) / 255.0
            sample = {'image': image}
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image1 = sample['image']
            image1 = to_tensor(image1)
            image1 = normalize(image1)

            return image1

        # 数据集长度
        def __len__(self):
            return len(self.data)




    image_path = '/dataset/kitti/data/test.csv.video.bak'        
    a=MyDataset(image_path)
    loader = DataLoader(a, 1,
                 shuffle = False,
                 num_workers=1,
                 pin_memory=False,
                 sampler=None)

    #sample = ToTensor(sample)
    #model
    #model = DepthLitModel.load_from_checkpoint(config.CKPT_PATH, config=config)
    model = DepthLitModel.load_from_checkpoint(config.CKPT_PATH, config=config)
    trainer = Trainer(gpus=get_gpus(args), default_root_dir=config.OUTPUT_DIR)
    trainer.predict(model, loader)


    #out
    #拓展维度
    #image1=image1.unsqueeze(0)
    #output_dict=model.net(image1)
    #print(output_dict)
    #depth_output = output_dict["depth_final", 1]
