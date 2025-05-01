import os
import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from torchkit.util import AverageMeter, Timer
from torchkit.util import accuracy_dist
from torchkit.util import AllGather
from torchkit.loss import get_loss
from torchkit.task import BaseTask
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
sys.path.append('./dareblopy')


class TrainTask(BaseTask):
    """ TrainTask in distfc mode, which means classifier shards into multi workers
    """

    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)

    def loop_step(self, epoch):
        backbone, heads = self.backbone, list(self.heads.values())
        backbone.train()  # set to training mode
        for head in heads:
            head.train()

        batch_sizes = self.batch_sizes
        am_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        t = Timer()

        for step, samples in enumerate(self.train_loader):
            # Debug input data
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = samples[0].to(device, non_blocking=True)
            labels = samples[1].to(device, non_blocking=True)
            print(f"Step {step}: Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            print(f"Step {step}: Labels: {labels[:5]}")  # Print the first 5 labels

            if self.amp:
                with amp.autocast():
                    features = backbone(inputs)
                features = features.float()
            else:
                features = backbone(inputs)

            # Debug model outputs
            print(f"Step {step}: Features shape: {features.shape}")
            print(f"Step {step}: Features (first 5): {features[:5]}")

            # Gather features and labels
            features_gather = AllGather(features, self.world_size)
            labels_gather = AllGather(labels, self.world_size)

            losses = []
            for i in range(len(batch_sizes)):
                if self.pfc:
                    outputs, labels, original_outputs = heads[i](features_gather[i], labels_gather[i], head_opts[i])
                else:
                    outputs, labels, original_outputs = heads[i](features_gather[i], labels_gather[i])

                # Debug outputs and loss
                print(f"Step {step}, Head {i}: Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                loss = self.loss(outputs, labels) * self.branch_weights[i]
                print(f"Step {step}, Head {i}: Loss: {loss.item()}")
                losses.append(loss)

            # Compute total loss
            total_loss = sum(losses)
            print(f"Step {step}: Total loss: {total_loss.item()}")

            # Backward pass
            self.backbone_opt.zero_grad()
            for head_opt in self.head_opts:
                head_opt.zero_grad()

            if self.amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.backbone_opt)
                for head_opt in self.head_opts:
                    self.scaler.step(head_opt)
                self.scaler.update()
            else:
                total_loss.backward()

                # Debug gradients
                for name, param in self.backbone.named_parameters():
                    if param.grad is not None:
                        print(f"Step {step}: {name} gradient mean: {param.grad.abs().mean()}")
                    else:
                        print(f"Step {step}: {name} has no gradient")

                self.backbone_opt.step()
                for head_opt in self.head_opts:
                    head_opt.step()

            # Debug learning rate
            for param_group in self.backbone_opt.param_groups:
                print(f"Step {step}: Learning rate: {param_group['lr']}")

            # Log time cost
            cost = t.get_duration()
            self.update_log_buffer({'time_cost': cost})

            # call hook function after_train_iter
            self.call_hook("after_train_iter", step, epoch)

    def make_optimizers(self):
        # Optimizer for backbone
        self.opt = torch.optim.SGD(
            self.backbone.parameters(),
            lr=self.cfg['LRS'][0],
            momentum=self.cfg['MOMENTUM'],
            weight_decay=self.cfg['WEIGHT_DECAY']
        )
        self.backbone_opt = self.opt 

        # Optimizer(s) for heads
        all_head_params = []
        for head in self.heads.values():
            all_head_params += list(head.parameters())

        self.head_opts = [torch.optim.SGD(  
            all_head_params,
            lr=self.cfg['LRS'][0],
            momentum=self.cfg['MOMENTUM'],
            weight_decay=self.cfg['WEIGHT_DECAY']
        )]





        
        

    def prepare(self):
        """ common prepare task for training
        """
        for key in self.cfg:
            print(key, self.cfg[key])
        if self.cfg['METHOD'] == 'PartialFace':
            self.make_inputs(num_dupices=self.cfg['NUM_DUPS'])
            self.make_model(in_channels=27)
        else:
            self.make_inputs()
            self.make_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = get_loss('DistCrossEntropy').to(device)
        self.make_optimizers()
        self.register_hooks()
        self.pfc = self.cfg['HEAD_NAME'] == 'PartialFC'

    def train(self):
        """
        make inputs
            |
        make model
            |
        make loss function
            |
        make optimizer
            |
        make auto mix precision grad scalar
            |
        register hooks
            |
        Distributed Data Parallel mode
            |
        loop_step
        """
        self.prepare()
        self.call_hook("before_run")
        if torch.cuda.is_available():
            self.backbone = DDP(self.backbone.to(f"cuda:{self.local_rank}"), device_ids=[self.local_rank])
        else:
            self.backbone = DDP(self.backbone)  # No device_ids for CPU
        for epoch in range(self.start_epoch, self.epoch_num):
            self.call_hook("before_train_epoch", epoch)
            self.loop_step(epoch)
            self.call_hook("after_train_epoch", epoch)
        self.call_hook("after_run")


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(os.path.join(task_dir, 'train.yaml'))
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
