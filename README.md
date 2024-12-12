# Variational Autoencoder (VAE) with U-Net Integration

This repository features a **Variational Autoencoder (VAE)** architecture integrated with a U-Net model, designed for image and text processing tasks. The project uses advanced methods like cross-attention and classifier-free guidance to achieve high-quality outputs.

---

## Key Components

### VAE Encoder
- Reduces the dimensionality of input features while enriching their representation.
- Utilizes **Residual Blocks**, which consist of normalization and convolution layers for enhanced feature extraction.

### VAE Decoder
- Reconstructs the input data from the latent space created by the encoder.

---

## Main Architecture

### Overview
- The model incorporates a **VAE** for images and a **text encoder** (using a **CLIP encoder**) to embed text into a shared latent space.
- The embedded text guides the U-Net during generation tasks.

### U-Net Model
- Symmetric design:
  - **First Half**: Encoder reduces the size of the image.
  - **Second Half**: Decoder reconstructs the output by expanding the encoded features.
- **Enhancement**: Incorporates **prompt embeddings** from the text encoder using **cross-attention** to specify desired output characteristics.

### Time Embedding
- U-Net includes **time embeddings** to account for the noise level in the image at each step, ensuring accurate temporal noise reduction.

---

## Workflow

### Training
1. **Input**:
   - **Image**: Processed through the VAE Encoder.
   - **Text**: Embedded into the latent space using the CLIP encoder.
2. **Noise Prediction**:
   - The U-Net processes the image, text, and time embedding to predict noise.
3. **Denoising**:
   - The U-Net collaborates with the scheduler to iteratively remove noise from the image.
4. **Output**:
   - The final latent representation is passed to the VAE Decoder to reconstruct the image.

### Inference
- Demonstrates how the scheduler and U-Net collaborate:
  1. Noise is detected and reduced iteratively by the U-Net and scheduler.
  2. The process repeats until the image is fully denoised.
  3. The clean latent representation is passed to the decoder to produce the output.

---

## Classifier-Free Guidance

To improve output quality, **Classifier-Free Guidance** is applied using the formula:

\[
\text{output} = w \times (\text{output_conditioned} - \text{output_unconditioned}) + \text{output_unconditioned}
\]

### How It Works
- **Two Inference Passes**:
  1. One with the prompt (conditioned).
  2. One without the prompt (unconditioned).
- Combines outputs to generate results aligned with the input prompt.

---

## Example: Text-to-Image

In a text-to-image generation task:
1. Noise is added to the input image.
2. The U-Net predicts and removes noise at each time step.
3. The scheduler coordinates the denoising process by iteratively querying the U-Net.
4. The denoised latent representation is decoded into the final image.

---

## Files and Structure

- **`vae_encoder.py`**: Implementation of the VAE Encoder.
- **`vae_residual_block.py`**: Residual block with normalization and convolution layers used in the encoder.
- **`u_net.py`**: U-Net architecture with integrated cross-attention.
- **`scheduler.py`**: Defines the noise scheduling process.
- **`pipeline.py`**: Full pipeline for training and inference, showcasing how the scheduler and U-Net interact.

---

This repository provides a robust implementation of VAE with U-Net integration, supporting image and text-to-image generation tasks. Feedback is welcome!






Special thanks to the following repostiories
https://github.com/hkproj/pytorch-stable-diffusion.git