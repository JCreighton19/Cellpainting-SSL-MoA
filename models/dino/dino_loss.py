import torch
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    def __init__(self, proj_dim=256, ncrops=2, warmup_teacher_temp=0.04,
                 teacher_temp=0.1, warmup_epochs=10, nepochs=100,
                 center_momentum=0.95):
        super().__init__()
        self.ncrops = ncrops
        self.center_momentum = center_momentum
        self.teacher_temp = teacher_temp
        self.register_buffer("center", torch.zeros(1, proj_dim))

    def forward(self, student_out, teacher_out):
        """
        student_out: (B, proj_dim)
        teacher_out: (B, proj_dim)
        """
        # student log softmax
        student_temp = 0.2
        student_log_probs = F.log_softmax(student_out / student_temp, dim=-1)

        # update center (EMA of teacher outputs)
        teacher_logits = (teacher_out - self.center) / self.teacher_temp
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