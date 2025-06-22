from torch.utils import data as data
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    recursive_glob)
from basicsr.data.event_util import events_to_voxel_grid, events_to_voxel_grid_pytorch, voxel_norm
from basicsr.data.transforms import augment, triple_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, voxel2voxeltensor, padding, get_root_logger
from torch.utils.data.dataloader import default_collate

# 从read_data.py中导入必要的函数
from read_data import read_aps, read_evs, process_isp


class GoProSingleImageEventDataset(data.Dataset):
    """GoPro dataset for training recurrent networks for blurry image interpolation.
    """

    def __init__(self, opt):
        super(GoProSingleImageEventDataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.m = opt['num_end_interpolation']
        self.n = opt['num_inter_interpolation']
        self.num_input_blur = 2  # 保留参数，实际单图像模式中只使用1张
        self.num_input_gt = 2*self.m + self.n  # 保留参数，实际单图像模式中只使用1张
        self.num_bins = opt['num_bins']
        self.split = opt['phase']  # 'train' 或 'test'
        self.norm_voxel = opt.get('norm_voxel', True)
        self.one_voxel_flg = opt.get('one_voxel_flag', True)
        self.return_deblur_voxel = opt.get('return_deblur_voxel', False)
        self.return_deblur_voxel = self.return_deblur_voxel and self.one_voxel_flg
        
        # 新增参数：训练集比例
        self.train_ratio = opt.get('train_ratio', 0.8)  # 默认80%作为训练集
        self.random_seed = opt.get('random_seed', 42)  # 随机种子，保证分割可复现

        # 初始化路径列表
        self.allBlurPath = []  # 模糊图像路径 (aps_raw_int10)
        self.allSharpPath = []  # 清晰图像路径 (gt_raw_int10)
        self.alleventSeqsPath = []  # 事件序列路径 (events)

        # 检查目录是否存在
        blur_folder = os.path.join(self.dataroot, 'aps_raw_int10')
        gt_folder = os.path.join(self.dataroot, 'gt_raw_int10')
        event_folder = os.path.join(self.dataroot, 'events')
        
        for folder in [blur_folder, gt_folder, event_folder]:
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Data directory not found: {folder}")

        # 获取所有文件索引
        file_indices = sorted([
            os.path.splitext(f)[0] 
            for f in os.listdir(blur_folder) 
            if f.endswith('.bin')
        ])

        # 分割数据集
        if self.split in ['train', 'test']:
            # 使用sklearn的train_test_split进行分割
            train_indices, test_indices = train_test_split(
                file_indices, 
                train_size=self.train_ratio,
                random_state=self.random_seed,
                shuffle=True
            )
            
            # 根据当前split选择使用的索引
            indices = train_indices if self.split == 'train' else test_indices
        else:
            # 如果split不是train或test，则使用全部数据
            indices = file_indices

        # 构建路径列表
        for idx in indices:
            blur_path = os.path.join(blur_folder, f"{idx}.bin")
            gt_path = os.path.join(gt_folder, f"{idx}.bin")
            
            # 构建事件序列路径
            int_idx = int(idx)
            event_indices = [f"{i:04d}" for i in range(int_idx - 5, int_idx + 6)]  # 前后各5个文件
            event_paths = [os.path.join(event_folder, f"{e_idx}.bin") for e_idx in event_indices]
            
            # 检查事件文件是否存在
            valid_event_paths = [p for p in event_paths if os.path.exists(p)]
            
            self.allBlurPath.append(blur_path)
            self.allSharpPath.append(gt_path)
            self.alleventSeqsPath.append(valid_event_paths)

        # 验证路径数量匹配
        assert len(self.allBlurPath) == len(self.allSharpPath) == len(self.alleventSeqsPath), \
            f'The number of blur/sharp/event: {len(self.allBlurPath)}/{len(self.allSharpPath)}/{len(self.alleventSeqsPath)} does not match.'

        # 文件客户端
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # 时间增强配置
        self.random_reverse = opt.get('random_reverse', False)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation: random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        # 其余部分保持不变...
        # （与之前提供的代码相同）
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        # 获取路径
        image_path = self.allBlurPath[index]  # 模糊图像路径 (aps_raw_int10)
        gt_path = self.allSharpPath[index]  # 清晰图像路径 (gt_raw_int10)
        event_paths = self.alleventSeqsPath[index]  # 事件序列路径 (events)

        # 读取模糊图像
        img_lq = read_aps(image_path)
        img_lq = process_isp(img_lq, self.opt['isp_conf'])

        # 读取清晰图像
        img_gt = read_aps(gt_path)
        img_gt = process_isp(img_gt, self.opt['isp_conf'])

        h_lq, w_lq, _ = img_lq.shape

        # 读取事件并转换为体素网格
        all_quad_event_array = np.zeros((0, 4)).astype(np.float32)
        for event_path in event_paths:
            if not os.path.exists(event_path):
                continue
                
            evs = read_evs(event_path)
            # 确保事件数据格式正确：(t, x, y, p)
            t = evs[:, 0].astype(np.float32)[:, np.newaxis]  # 时间戳
            x = evs[:, 1].astype(np.float32)[:, np.newaxis]  # x坐标
            y = evs[:, 2].astype(np.float32)[:, np.newaxis]  # y坐标
            p = evs[:, 3].astype(np.float32)[:, np.newaxis]  # 极性
            this_quad_event_array = np.concatenate((t, x, y, p), axis=1)  # N,4
            all_quad_event_array = np.concatenate((all_quad_event_array, this_quad_event_array), axis=0)

        # 转换为体素网格
        voxel = events_to_voxel_grid(
            all_quad_event_array, 
            num_bins=self.num_bins, 
            width=w_lq, 
            height=h_lq, 
            return_format='HWC'
        )
        
        # 体素归一化
        if self.norm_voxel:
            voxel = voxel_norm(voxel)

        # 随机裁剪
        if gt_size is not None:
            img_gt, img_lq, voxel = triple_random_crop(img_gt, img_lq, voxel, gt_size, scale, gt_path)
        
        # 数据增强
        all_imgs = [img_gt, img_lq, voxel]
        img_results = augment(all_imgs, self.opt['use_hflip'], self.opt['use_rot'])

        # 转换为张量
        img_results = img2tensor(img_results)  # hwc -> chw
        img_gt = img_results[0]
        img_lq = img_results[1]
        voxel = img_results[2]

        # 获取序列和索引信息
        seq = os.path.basename(self.dataroot)
        origin_index = os.path.basename(image_path).split('.')[0]

        return {'lq': img_lq, 'gt': img_gt, 'voxel': voxel, 'seq': seq, 'origin_index': origin_index}

    def __len__(self):
        return len(self.allBlurPath)
