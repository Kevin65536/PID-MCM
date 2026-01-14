# Continous data tokenizer

## FSQ and VQ-VAE

### FSQ

FSQ stands for "Fixed Scalar Quantization".

- **Mechanism**: It projects the continuous VAE representation into a low-dimensional space (typically less than 10 dimensions). Each dimension is then quantized to a small, fixed set of predefined scalar values (e.g., in the range \([-1,1]\) with a few equidistant values). The final "code" is the combination of these scalar levels across dimensions, forming an implicit codebook.

- **Key Feature**: The codebook is a fixed, predefined grid, not a set of learnable parameters that need active management during training.

- **Advantage**: This design avoids common issues associated with VQ, such as codebook collapse, and eliminates the need for complex auxiliary losses (like commitment loss or codebook reseeding). It scales well with larger implicit codebook sizes and achieves high codebook utilization.

### VQ-VAE

VQ-VAE stands for "Vector Quantized Variational Autoencoder".

- **Mechanism**: It uses a learnable codebook where each code is a vector in the latent space. During training, the continuous latent representations produced by the encoder are matched to the nearest code in the codebook, and the corresponding code index is used for reconstruction.
  - **Encoder**: A neural network takes continuous input data and  compresses it into a grid of continuous vectors (e.g., a 32*32 grid of 64-dimensional vectors).
  - **Codebook**: A set of learnable vectors (codes) that the encoder's output vectors are quantized against.
  - **Quantization**: Each continuous vector from the encoder is replaced by the nearest code from the codebook, and the index of that code is stored as the discrete representation.
  - **Decoder**: The decoder reconstructs the original data from the quantized codes.

- **Key Feature**: The codebook is learned during training, allowing the model to adaptively find useful representations.

- **Challenges**: VQ-VAE can suffer from codebook collapse, where only a few codes are used frequently while others are ignored. This requires additional techniques like commitment loss or codebook reseeding to maintain codebook diversity.
