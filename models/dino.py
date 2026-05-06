# models/dino.py
import torch
import torch.nn as nn
import timm

class CellPaintingViT(nn.Module):
    """
    DINOv2 ViT-S with modified patch embedding to accept 5-channel input.
    """
    def __init__(self, in_channels=5, img_size=224, pretrained=True):
        super().__init__()

        # load pretrained DINOv2 ViT-S
        self.vit = timm.create_model(
            'vit_small_patch14_dinov2',
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0  # remove classification head to just get embeddings
        )

        # replace patch embedding to accept in_channels instead of 3
        old_proj = self.vit.patch_embed.proj  # Conv2d(3, embed_dim, kernel_size=14, stride=14)
        new_proj = nn.Conv2d(
            in_channels,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None
        )

        # initialize new projection weights
        with torch.no_grad():
            # copy pretrained RGB weights for first 3 channels
            new_proj.weight[:, :3].copy_(old_proj.weight)

            # initialize extra channels as mean of RGB
            if in_channels > 3:
                mean_weight = old_proj.weight.mean(dim=1, keepdim=True)
                for i in range(3, in_channels):
                    new_proj.weight[:, i:i + 1, :, :] = mean_weight

            if old_proj.bias is not None:
                new_proj.bias.data.copy_(old_proj.bias.data)

        self.vit.patch_embed.proj = new_proj

    def forward(self, x):
        # x: (B, 5, 224, 224)
        return self.vit(x)  # (B, embed_dim) — 384 for ViT-S


if __name__ == "__main__":
    model = CellPaintingViT(in_channels=5, pretrained=True)
    x = torch.randn(2, 5, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)  # should be (2, 384)
    print("Num parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")