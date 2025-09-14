# 3D U-Net Architecture for Brain Tumor Segmentation

This file contains a **3D U-Net model** for multi-class brain tumor segmentation.

## Overview

- Input: 3D MRI volume `(D,H,W,C)`  
- Output: 3D segmentation mask `(D,H,W,NUM_CLASSES)`  
- Encoder: compresses spatial info, extracts features  
- Bottleneck: dense feature representation at smallest spatial size  
- Decoder: upsamples, combines features with skip connections  

## Mathematical Formulation

1. **3D Convolution**

```math
F_{out}^{(c)}(x,y,z) = \sigma \Big( \sum_{i=1}^{C_{in}} \sum_{u=0}^{k_d-1} \sum_{v=0}^{k_h-1} \sum_{w=0}^{k_w-1} K^{(c,i)}_{u,v,w} \cdot F_{in}^{(i)}(x+u, y+v, z+w) + b^{(c)} \Big)
```

- `F_{in}^{(i)}` = input channel ```i``` 
- ```K^{(c,i)}``` = kernel for output channel ```c```, input channel ```i```
- ```\sigma``` = activation function (ReLU)  

2. **MaxPooling3D**

```math
O(x,y,z) = \max_{(u,v,w) \in \text{pool\_size}} F_{in}(x+u, y+v, z+w)
```

- Reduces spatial dimensions while keeping feature depth intact  

3. **Upsampling + Skip Connections**

```math
U_{l} = \text{concat}(\text{Upsample}(F_{l+1}), F_{skip})
```

- Combines encoder features with decoder upsampled features for precise localization  

4. **Output Layer**

```math
\hat{Y}(x,y,z,c) = \text{softmax}_c(F_{final}(x,y,z))
```

- Predicts **voxel-wise class probabilities**  

---

### Why U-Net is effective

- Multiple conv layers → increasing **receptive field**  
- Skip connections → preserve **fine spatial details**  
- Bottleneck → compresses global features for efficient learning  
- Softmax output → supports **multi-class segmentation**

