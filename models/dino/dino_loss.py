import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist


class DINOLoss(nn.Module):
    def __init__(self, proj_dim=4096, ncrops=2, warmup_teacher_temp=0.01,
                 teacher_temp=0.04, warmup_epochs=30, nepochs=100,
                 center_momentum=0.99):
        super().__init__()
        self.ncrops = ncrops
        self.center_momentum = center_momentum
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_epochs = warmup_epochs
        self.register_buffer("center", torch.zeros(1, proj_dim))


    def forward(self, student_out, teacher_out, epoch=0):
        # temperature schedule
        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            teacher_temp = self.warmup_teacher_temp + alpha * (
                    self.teacher_temp - self.warmup_teacher_temp
            )
        else:
            teacher_temp = self.teacher_temp

        student_temp = 0.1

        # teacher distribution - should be raw unnormalized logits!
        teacher_logits = (teacher_out - self.center) / teacher_temp
        teacher_probs = F.softmax(teacher_logits, dim=-1).detach()
        teacher_probs = teacher_probs.clamp(min=1e-8)

        # student distribution
        student_log_probs = F.log_softmax(student_out / student_temp, dim=-1)
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out, momentum=None):
        batch_center = teacher_out.mean(dim=0, keepdim=True) # local GPU mean
        momentum = (momentum if momentum is not None else self.center_momentum)

        # synchronize across GPUs
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center,op=dist.ReduceOp.SUM)
            batch_center /= dist.get_world_size()

        # EMA update
        self.center.mul_(momentum)
        self.center.add_(batch_center *(1.0 - momentum))