import torch
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    def __init__(self, proj_dim=8192, ncrops=2, warmup_teacher_temp=0.04,
                 teacher_temp=0.1, warmup_epochs=10,
                 center_momentum=0.95):
        super().__init__()
        self.ncrops = ncrops
        self.center_momentum = center_momentum
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_epochs = warmup_epochs
        self.register_buffer("center", torch.zeros(1, proj_dim))

    def get_teacher_temp(self, epoch):
        if epoch >= self.warmup_epochs:
            return self.teacher_temp

        # linear warmup from warmup_teacher_temp → teacher_temp
        alpha = epoch / self.warmup_epochs
        return self.warmup_teacher_temp + alpha * (self.teacher_temp - self.warmup_teacher_temp)

    def forward(self, student_out, teacher_out, epoch=0):
        """
        student_out: (B, proj_dim)
        teacher_out: (B, proj_dim)
        """
        # student log softmax
        student_temp = 0.2
        student_log_probs = F.log_softmax(student_out / student_temp, dim=-1)

        # update center (EMA of teacher outputs)
        temp = self.get_teacher_temp(epoch)
        teacher_logits = (teacher_out - self.center) / temp
        teacher_probs = F.softmax(teacher_logits, dim=-1).detach()

        # cross entropy loss
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()

        self.update_center(teacher_out)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum)
        self.center.add_(batch_center * (1 - self.center_momentum))