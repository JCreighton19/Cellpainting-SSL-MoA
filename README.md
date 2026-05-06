The biological problem
When researchers image cells, they want to answer questions like: did this drug do anything? are these cells sick or healthy? what stage of division is this cell in? Traditionally, answering those questions requires either:
A human expert looking at images and labeling them (slow, expensive, subjective), or
Fluorescent tags that chemically mark specific proteins so you can "see" them (requires lab work, can kill or alter the cells, and you can only look at what you tagged)
The dream is: feed a model raw, unlabeled cell images and have it learn a rich enough internal representation that you can answer biological questions just by querying that representation — no labels needed during training.

What SSL is doing here
Self-supervised learning sidesteps the labeling problem by inventing its own supervisory signal from the data itself. The three methods worth knowing:
DINO / DINOv2 — trains a model by showing it two differently-augmented crops of the same image and forcing the representations to match. The key insight is it uses a "teacher-student" setup where the teacher is an exponential moving average of the student, which prevents collapse. DINOv2 adds curated data and regularization. What's remarkable is that DINO features, with zero labels, often produce segmentation maps that align with semantic boundaries — in natural images it "finds" objects. The open question is whether it finds biologically meaningful structure in microscopy.
MAE (Masked Autoencoder) — masks out 75% of image patches and trains the model to reconstruct them. Forces the model to build a global understanding of image structure. Tends to learn more texture/structure-sensitive features than DINO.
The key difference for bio: DINO features tend to cluster by object identity (this cell vs. that cell), while MAE features tend to capture texture and morphology more. For biology, morphology is often what you care about — a cell treated with a cytoskeleton-disrupting drug looks physically different even if you can't tag anything.

Why JUMP-CP specifically
JUMP-CP (Joint Undertaking in Morphological Profiling - Cell Painting) is a dataset released in 2023 by the Broad Institute and Recursion. Some context:
~140,000 chemical and genetic perturbations applied to cells
Images taken with Cell Painting, which uses 6 fluorescent dyes that non-specifically stain cell compartments (nucleus, mitochondria, ER, cytoskeleton, etc.) — not targeted to specific proteins
~116 million individual cell images total
Fully public on AWS S3
The reason it's interesting for SSL: the perturbation labels exist but you deliberately don't use them during training. After training, you check whether your learned embedding space organizes drugs by their known mechanism of action — do all the drugs that disrupt microtubules cluster together? If yes, your model has learned something biologically real.
This evaluation approach (called "perturbation retrieval" or "mechanism of action prediction") is well-defined, has existing benchmarks, and gives you a clean story: I trained a model with no labels and it recovered known drug biology.

What's actually novel about doing this now
The existing work in this space mostly uses:
Older contrastive methods (SimCLR, MoCo) from 2020-2021
ResNet-50 backbones
A specific Cell Painting preprocessing pipeline called CellProfiler that extracts hand-crafted features first
What hasn't been done cleanly in public:
DINOv2 ViT backbone trained end-to-end directly on raw JUMP-CP images (not CellProfiler features)
Systematic comparison of DINO vs MAE features on biological retrieval tasks
Probing which attention heads in a DINO ViT correspond to which cell compartments
That last one is especially interesting — in natural image DINO, you can visualize attention maps and see the model attending to foreground objects. In Cell Painting, the analogous question is: does a head learn to attend to the nucleus? The mitochondria? This would be a genuinely new analysis.

Concrete project structure
The scope that's doable solo in 2-3 months:
Data pipeline — Download a manageable JUMP-CP subset (one cell line, one plate type — still millions of images). Write a PyTorch Dataset that loads 5-channel Cell Painting images. The non-obvious part: Cell Painting images aren't RGB, they're 5 separate grayscale channels. You need to decide how to handle this (train a 5-channel ViT, or project to 3 channels, or train one channel at a time).


Training — Fine-tune a pretrained DINOv2 ViT-S on your microscopy data. Don't train from scratch — you don't have the compute. The interesting question is how much the natural-image pretraining helps or hurts for this domain.


Evaluation — For each perturbation in a held-out set, embed all images of that perturbation, average the embeddings, then ask: can you retrieve perturbations with the same MoA using cosine similarity? Compare against CellProfiler baseline features. This is a clean, quantitative benchmark.


Attention visualization — Pull DINO attention maps for a sample of images and correlate them with the known channel stains. This is the novel analysis piece.


Release — Clean repo, a Weights & Biases training log, and a Jupyter notebook showing the biological analysis. Optionally a HuggingFace model card.



The reason this is strong for an ML engineer role specifically: it requires real engineering (multi-channel image loading, distributed training considerations, embedding-scale retrieval) and biological interpretation. Most candidates can do one or the other. The combination is what biotech ML teams actually need.

The case for transfer learning here
DINOv2 was pretrained on ~142 million natural images (LVD-142M). Microscopy images are obviously a different domain — no backgrounds, no objects in the traditional sense, weird 5-channel input instead of RGB. So the question is: does that natural image pretraining help or hurt?
The answer from the literature is consistently: it helps, even across very different domains. The low-level features ViTs learn — edges, textures, local structure, spatial relationships between patches — transfer surprisingly well to microscopy. A nucleus boundary is still an edge. Texture differences between cytoplasm and organelles are still texture differences. The model doesn't need to know what a mitochondrion is to have useful low-level features for distinguishing it.
Practically for you: training a ViT from scratch on JUMP-CP would require weeks of compute on many GPUs. Fine-tuning a pretrained DINOv2 is feasible on a single GPU in days, possibly hours depending on how much data you use.

The 5-channel problem
This is the main technical wrinkle with transfer learning here. DINOv2 expects 3-channel RGB input. Your Cell Painting images are 5-channel. You have a few options:
Option 1: Projection layer Add a learned Conv2d(5, 3, kernel_size=1) before the ViT patch embedding. This maps your 5 channels to 3, then everything downstream is unchanged. Simple, clean, keeps all pretrained weights intact. The downside is you're immediately compressing channel information before the model even sees it.
Option 2: Modify the patch embedding layer The first layer of a ViT is a patch embedding — essentially a Conv2d that maps (3, patch_size, patch_size) → embedding dim. You can reinitialize this to accept 5 channels, and initialize the weights for the first 3 channels from the pretrained model and randomly initialize the remaining 2. The rest of the network stays pretrained. This is more principled — the model sees all 5 channels directly.
Option 3: Train on each channel separately Treat each channel as a grayscale image, tile it to 3 channels (just repeat it), and run them through separately. Average or concatenate the embeddings. Loses cross-channel relationships but is the easiest to implement and closest to the pretrained distribution.
For a resume project, Option 2 is the most defensible and interesting choice. It's what a good ML engineer would actually do and it shows you understand the architecture.

What you're actually fine-tuning
You have a spectrum of choices:
Approach
What's frozen
What's trained
Compute
Linear probe only
Entire ViT
One linear layer on top
Trivial
Partial fine-tune
Early ViT layers
Last N transformer blocks + patch embed
Moderate
Full fine-tune
Nothing
Everything
Heavy
DINO objective on top
Pretrained weights as init
Everything, with SSL loss
Heavy but principled

The most interesting approach for your project is the last one — use DINOv2 weights as initialization, then continue DINO training on your microscopy data. This is called continued pretraining or domain-adaptive pretraining. The model starts with strong general visual features and adapts them to the microscopy domain. You get the best of both worlds.
The linear probe is useful as a baseline and evaluation tool regardless of which training approach you pick.

The novel angle this opens up
Because you're starting from a pretrained checkpoint and fine-tuning on microscopy, you can do an analysis that hasn't been done cleanly in public:
How much does domain-adaptive pretraining actually change what the model knows?
Concretely — take the pretrained DINOv2 checkpoint, take your fine-tuned checkpoint, and compare their attention maps and embedding spaces on the same cell images. Which biological features emerge only after fine-tuning? Which were already captured by natural image pretraining? This is a genuinely interesting empirical question and a strong narrative for a writeup or blog post.

The short answer: yes, transfer learning makes sense, the main engineering challenge is the 5-channel input, and continued DINO pretraining from a DINOv2 init is the most defensible and novel approach. Want to get into what the actual training code skeleton would look like?

File structure
jump-dino/
│
├── data/
│   ├── download.py          # script to pull from JUMP-CP S3
│   ├── preprocess.py        # normalization, cell cropping
│   └── README.md            # documents exactly what data subset you used
│
├── datasets/
│   └── cellpainting.py      # PyTorch Dataset class for 5-channel images
│
├── models/
│   ├── dino.py              # DINO training wrapper / loss
│   ├── vit.py               # modified ViT with 5-channel patch embed
│   └── projection.py        # optional 5→3 channel projection layer
│
├── train.py                 # main training script
├── evaluate.py              # MoA retrieval, linear probe evaluation
├── visualize_attention.py   # attention map analysis
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb   # UMAP, clustering
│   └── 03_attention_maps.ipynb       # the novel analysis piece
│
├── configs/
│   └── default.yaml         # hyperparameters, paths, model config
│
├── scripts/
│   └── train.sh             # SLURM job script for the cluster
│
├── requirements.txt
└── README.md


Raw Dataset
Source → batch → plate → well → image

Images are of the format: r01c01f01p01-ch1sk1fk1fl1.tiff
r01c01 — row 1, column 1 (the well)
f01 — field of view 1 (site within the well)
p01 — plane (z-plane, likely just 1 for standard Cell Painting)
ch1 through ch5 — the 5 fluorescence channels
72 files ÷ 5 channels = 14 fields of view in this well
