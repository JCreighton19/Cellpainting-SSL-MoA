import torch
import torch.nn.functional as F
from torch import nn

class DINOLoss(nn.Module):
    def __init__(self, proj_dim=256, ncrops=2, warmup_teacher_temp=0.04,
                 teacher_temp=0.1, warmup_epochs=10, nepochs=100,
                 center_momentum=0.99):
        super().__init__()
        self.ncrops = ncrops
        self.center_momentum = center_momentum
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_epochs = warmup_epochs
        self.register_buffer("center", torch.zeros(1, proj_dim))

    def forward(self, student_out, teacher_out, epoch=0, update_center=False):
        """
        student_out: (B, D)  — one student view
        teacher_out: (B, D)  — one teacher view (cross-view pair)
        update_center: should be False for all but the explicit once-per-step center update
        """
        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            teacher_temp = self.warmup_teacher_temp + alpha * (self.teacher_temp - self.warmup_teacher_temp)
        else:
            teacher_temp = self.teacher_temp

        student_temp = 0.1
        teacher_probs = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1).detach()
        student_log_probs = F.log_softmax(student_out / student_temp, dim=-1)
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        if update_center:
            self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum)
        self.center.add_(batch_center * (1 - self.center_momentum))