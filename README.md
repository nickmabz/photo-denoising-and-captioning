# Photo denoising and captioning

An image-processing pipeline for scanned photographs: **classify** what is a photo, **denoise** it with a convolutional auto-encoder, then **caption** it with an attention-based RNN. Built with TensorFlow/Keras as a group project for the Data Science course at CESI École d'Ingénieurs (Lyon).

```
scanned image ──► 1. image-type classifier (keep photos) ──► 2. denoiser (residual U-Net) ──► 3. captioning (CNN + attention + RNN) ──► caption
```

## Repository contents

| Notebook | What it does |
|---|---|
| `01_image_classification.ipynb` | 5-class image classification (Painting, Photo, Schematics, Sketch, Text) with a baseline CNN and a CNN with dropout. |
| `model_with_dropout.weights.h5` | Saved weights of the classification model with dropout. |
| `02_image_denoising.ipynb` | Residual U-Net denoising auto-encoder, compared with Gaussian, median and Non-Local Means filters. |
| `03_image_captioning.ipynb` | Image captioning on COCO: a baseline model and an attention model, evaluated with BLEU (greedy vs beam search). |

## 1. Image classification

- **Goal:** classify images into 5 classes (Painting, Photo, Schematics, Sketch, Text) so that only photographs go on to the denoising and captioning steps.
- **Data:** 41,398 images in 5 folders, 80 / 20 train / validation split, images resized to 180×180.
- **Models** (small CNNs, trained from scratch, 20 epochs, Adam, sparse categorical cross-entropy, batch size 16):
  - baseline: random horizontal flip → 3 × (Conv + MaxPooling) with 16, 32, 64 filters → Dense(80) → softmax;
  - same network with **Dropout (0.2)** before the dense layers (weights saved in `model_with_dropout.weights.h5`).

## 2. Image denoising

**Problem.** 148 colour photographs are provided as "noisy", with no clean version. We treat the photos as targets and add synthetic Gaussian noise to build the training inputs (self-supervised).

**Data preparation.** Pixels scaled to [0, 1]; each photo resized to 128×128 (aspect ratio kept, borders reflect-padded); split by photo into 118 training and 30 held-out images; Gaussian noise (σ ≈ 0.06) added to the inputs.

**Model.** A residual U-Net auto-encoder (~2.06 M parameters):
- encoder: 3 × (residual block + max-pooling), filters 32 → 64 → 128, down to a 16×16×256 bottleneck;
- decoder: 3 × (bilinear up-sampling + skip connection + residual block);
- output: a 1×1 convolution predicts a correction added to the noisy input, followed by a sigmoid.

**Training.** Loss = 0.8 · MSE + 0.2 · (1 − SSIM) (MSE keeps pixel values and colours, SSIM keeps edges and local structure); Adam, batch size 16, 50 epochs.

**Results** (held-out photos, synthetic noise):

| | Train | Validation |
|---|---|---|
| Loss | 0.0210 | 0.0226 |
| PSNR | 29.2 dB | 28.3 dB |
| SSIM | 0.900 | 0.893 |

Train and validation curves stay close, so there is no sign of overfitting. Trained at a single noise level, the model improves PSNR over the noisy input at every level tested (about +1.4 dB at light noise, +5 to +6 dB at stronger noise) and beats Non-Local Means in 3 of 4 cases.

**Limitations.** The targets are not truly clean, the noise is synthetic Gaussian (real noise differs), and the model was trained at one noise level. The 30 held-out photos serve as both validation and test set.

## 3. Image captioning

- **Data:** COCO captions, loaded with HuggingFace `datasets` (Karpathy split).
- **Image features:** pre-trained ResNet50 (7×7 grid of feature vectors).
- **Text:** tokenised captions, padded, padding ignored in the loss.
- **Model 1:** average-pooled image embedding + LSTM decoder.
- **Model 2:** attention over the 49 image regions + LSTM decoder; attention maps are visualised.
- **Decoding:** greedy search vs beam search.

**Best result: Model 2 with beam search**

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|
| 0.636 | 0.440 | 0.296 | 0.198 |

Beam search improved BLEU-3 and BLEU-4 for both models. Model 1 tends to produce generic captions; Model 2 follows the image content better but sometimes repeats words and hallucinates objects. Roadmap in the notebook: stronger visual backbone (ViT / CLIP), repetition penalty, sub-word tokenisation, and metrics beyond BLEU (METEOR, CIDEr).

## Running the notebooks

- Python 3, TensorFlow / Keras, NumPy, Matplotlib, scikit-image, OpenCV, Pillow, pandas, scikit-learn, HuggingFace `datasets`, NLTK.
- The classification notebook expects the image folders in `dataset/` (not included) and uses a cache folder: change the `.cache(...)` path to one that exists on your machine.
- The denoising notebook expects the 148 photos in a `Dataset/` folder next to it (not included in this repository).
- The captioning notebook downloads its data from HuggingFace; a GPU is recommended.

CESI École d'Ingénieurs, Data Science & AI, 2026.
