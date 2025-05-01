import os
import sys
import logging
from collections import OrderedDict
import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from ..backbone import get_model
from ..head import get_head
from ..hooks import CheckpointHook, LogHook, SummaryHook, LearningRateHook
from ..util import load_config, get_class_split, separate_resnet_bn_paras, CkptLoader, CkptSaver
from ..data import MultiDataset, MultiDistributedSampler

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s: %(message)s')


def get_fixed_transforms(rgb_mean, rgb_std):
    before_crop_transform = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])

    crop_transform = transforms.RandomResizedCrop(
        size=(112, 112),
        scale=(0.2, 1.0),
        ratio=(0.75, 1.3333333333333333)
    )

    after_crop_transform = transforms.Compose([
        transforms.Resize((112, 112)),  # safe to keep this for consistency
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ])

    return [before_crop_transform, crop_transform, after_crop_transform]


class BranchMetaInfo(object):
    def __init__(self, name, batch_size, weight=1.0, scale=64.0, margin=0.5):
        self.name = name
        self.batch_size = batch_size
        self.weight = weight
        self.scale = scale
        self.margin = margin


class BaseTask(object):
    def __init__(self, cfg_file):
        self.cfg = load_config(cfg_file)
        self.rank = 0
        self.local_rank = 0
        self.world_size = 0

        self.step_per_epoch = 0
        self.warmup_step = self.cfg['WARMUP_STEP']
        self.start_epoch = self.cfg['START_EPOCH']
        self.epoch_num = self.cfg['NUM_EPOCH']

        self.input_size = self.cfg['INPUT_SIZE']
        self.branches = OrderedDict()
        self.train_loader = None

        self.dist_fc = self.cfg.get('DIST_FC', True)
        self.amp = self.cfg.get('AMP', False)

        self.backbone = None
        self.heads = OrderedDict()
        self.summary = OrderedDict()
        self.log_buffer = OrderedDict()
        self.scaler = amp.GradScaler() if self.amp else None

        for branch in self.cfg['DATASETS']:
            meta = BranchMetaInfo(branch['name'], branch['batch_size'],
                                  branch.get('weight', 1.0),
                                  branch.get('scale', 64),
                                  branch.get('margin', 0.5))
            self.branches[meta.name] = meta
            logging.info("Dataset %s, batch_size %d, weight %f, scale %d, margin %f",
                         meta.name, meta.batch_size, meta.weight, meta.scale, meta.margin)

        self.batch_sizes = [b.batch_size for b in self.branches.values()]
        self.branch_weights = [b.weight for b in self.branches.values()]

    def init_env(self):
        seed = self.cfg['SEED']
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")

        dist.init_process_group(backend=self.cfg['DIST_BACKEND'], init_method=self.cfg["DIST_URL"])
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ['LOCAL_RANK'])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        logging.info("world_size: %d, rank: %d, local_rank: %d", self.world_size, self.rank, self.local_rank)
        self.cfg['WORLD_SIZE'] = self.world_size
        self.cfg['RANK'] = self.rank

    def register_hooks(self):
        self._hooks = [
            LogHook(100, self.rank),
            SummaryHook(self.cfg['LOG_ROOT'], 100, self.rank),
            CheckpointHook(self.cfg.get('SAVE_EPOCHS', list(range(1, self.cfg['NUM_EPOCH'] + 1))))
        ]

        lr_hook = LearningRateHook(self.cfg['LRS'], self.cfg['STAGES'], self.cfg['WARMUP_STEP'],
                                   recon=(self.cfg['METHOD'] == 'AdvFace'))
        self._hooks.append(lr_hook)

    def call_hook(self, fn_name, *args):
        for hook in self._hooks:
            getattr(hook, fn_name)(self, *args)

    def make_inputs(self, num_dupices=1):
        rgb_mean = self.cfg['RGB_MEAN']
        rgb_std = self.cfg['RGB_STD']
        transform = get_fixed_transforms(rgb_mean, rgb_std)
        ds_names = list(self.branches.keys())

        if self.cfg['METHOD'] == 'CASIA':
            from ..data import CASIADataset
            ds = CASIADataset(self.cfg['DATA_ROOT'], ds_names, transform,
                              num_duplices=num_dupices,
                              AdaFace_augment_prob=self.cfg['AdaFace_augment_prob'],
                              clean_txt=self.cfg['CASIA_clean_txt_path'])
        elif self.cfg['METHOD'] == 'TF-Synthetic':
            from ..data import TF_SyntheticDataset
            ds = TF_SyntheticDataset(self.cfg['DATA_ROOT'], ds_names, transform,
                                     num_duplices=num_dupices,
                                     AdaFace_augment_prob=self.cfg['AdaFace_augment_prob'])
        else:
            ds = MultiDataset(self.cfg['DATA_ROOT'], self.cfg['INDEX_ROOT'], ds_names, transform,
                              num_duplices=num_dupices)
            ds.make_dataset(shard=False)

        self.class_nums = ds.class_nums
        sampler = MultiDistributedSampler(ds, self.batch_sizes)
        self.train_loader = DataLoader(ds, sum(self.batch_sizes), shuffle=False,
                                       num_workers=self.cfg["NUM_WORKERS"], pin_memory=True,
                                       sampler=sampler, drop_last=False)
        self.step_per_epoch = len(self.train_loader)
        logging.info("Step_per_epoch = %d", self.step_per_epoch)

    def make_model(self, in_channels=3, task='FR'):
        method = self.cfg['METHOD']
        backbone_name = self.cfg['BACKBONE_NAME']
        self.backbone = get_model(backbone_name)(self.input_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        logging.info("%s Backbone Generated", backbone_name)

        embedding_size = self.cfg['EMBEDDING_SIZE']
        self.class_shards = []
        metric = get_head(self.cfg['HEAD_NAME'], dist_fc=self.dist_fc)

        for name, branch in self.branches.items():
            class_num = self.class_nums[name]
            class_shard = get_class_split(class_num, self.world_size)
            self.class_shards.append(class_shard)

            head = metric(in_features=embedding_size,
                        gpu_index=self.rank,
                        weight_init=torch.randn(embedding_size, class_num) * 0.01,
                        class_split=class_shard,
                        scale=branch.scale,
                        margin=branch.margin)
            self.heads[name] = head.to(self.device)
            logging.info('Split FC: %s', class_shard)


    def get_optimizer(self):
        backbone_bn, backbone_rest = separate_resnet_bn_paras(self.backbone)
        init_lr = self.cfg['LRS'][0]
        opt = {
            'backbone': optim.SGD([
                {'params': backbone_rest, 'weight_decay': self.cfg['WEIGHT_DECAY']},
                {'params': backbone_bn}
            ], lr=init_lr, momentum=self.cfg['MOMENTUM']),
            'heads': {name: optim.SGD([{'params': head.parameters()}], lr=init_lr,
                                      momentum=self.cfg['MOMENTUM'],
                                      weight_decay=self.cfg['WEIGHT_DECAY'])
                      for name, head in self.heads.items()}
        }
        return opt

    def update_log_buffer(self, vars):
        self.log_buffer.update(vars)

    def update_summary(self, vars):
        self.summary.update(vars)

    def save_ckpt(self, epoch):
        model_root = self.cfg['MODEL_ROOT']
        CkptSaver.save_backbone(self.backbone, model_root, epoch, self.rank)
        CkptSaver.save_heads(self.heads, model_root, epoch, self.dist_fc, self.rank)

        meta = {'EPOCH': epoch}
        if isinstance(self.opt, dict):
            meta['BACKBONE_OPT'] = self.opt['backbone'].state_dict()
        else:
            meta['OPTIMIZER'] = self.opt.state_dict()
        if self.amp:
            meta["AMP_SCALER"] = self.scaler.state_dict()
        CkptSaver.save_meta(meta, model_root, epoch, self.rank)
        logging.info("Save checkpoint at epoch %d ...", epoch)

    def load_pretrain_model(self):
        if self.cfg.get('BACKBONE_RESUME'):
            CkptLoader.load_backbone(self.backbone, self.cfg['BACKBONE_RESUME'], self.local_rank)
        if self.cfg.get('HEAD_RESUME'):
            CkptLoader.load_head(self.heads, self.cfg['HEAD_RESUME'], self.dist_fc, self.rank)
        if self.cfg.get('META_RESUME'):
            CkptLoader.load_meta(self.opt, self.scaler, self, self.cfg['META_RESUME'])

    def loop_step(self, epoch):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
