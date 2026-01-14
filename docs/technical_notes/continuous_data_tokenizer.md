# Continous data tokenizer

## FSQ and VQ-VAE

### FSQ

FSQ stands for "Fixed Scalar Quantization".

- **Mechanism**: It projects the continuous VAE representation into a low-dimensional space (typically less than 10 dimensions). Each dimension is then quantized to a small, fixed set of predefined scalar values (e.g., in the range \([-1,1]\) with a few equidistant values). The final "code" is the combination of these scalar levels across dimensions, forming an implicit codebook.

- **Key Feature**: The codebook is a fixed, predefined grid, not a set of learnable parameters that need active management during training.

- **Advantage**: This design avoids common issues associated with VQ, such as codebook collapse, and eliminates the need for complex auxiliary losses (like commitment loss or codebook reseeding). It scales well with larger implicit codebook sizes and achieves high codebook utilization.

### VQ-VAE



## Current EEG foundational model choices