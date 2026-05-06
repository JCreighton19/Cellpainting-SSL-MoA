import torch
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    def __init__(self, proj_dim=256, ncrops=2, warmup_teacher_temp=0.04,
                 teacher_temp=0.04, warmup_epochs=10, nepochs=100,
                 center_momentum=0.9):
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
        # sharpen teacher output
        teacher_out = F.softmax(
            (teacher_out - self.center) / self.teacher_temp, dim=-1
        ).detach()

        # student log softmax
        student_temp = 0.1
        student_out = F.log_softmax(student_out / student_temp, dim=-1)

        # cross entropy loss
        loss = -(teacher_out * student_out).sum(dim=-1).mean()

        # update center (EMA of teacher outputs)
        self.update_center(teacher_out)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = (self.center * self.center_momentum
                      + batch_center * (1 - self.center_momentum))