---
layout: single
author_profile: True
classes: wide
excerpt: "Revisiting View Selection in Neural Volume Rendering<br/>CVPR 2024"
header:
  overlay_image: /assets/images/Tessellation_MongeNet.png
  overlay_filter: 0.5
  caption: "Voronoi Tessellation for MongeNet sampled points.
  "
  actions:
    - label: "Paper"
      url: "https://github.com/wenwhx/nerfdirector"
    - label: "Code"
      url: "https://github.com/wenwhx/nerfdirector"
    - label: "Slides"
      url: "https://github.com/wenwhx/nerfdirector"
    - label: "Talk"
      url: "https://github.com/wenwhx/nerfdirector"
gallery_voronoi:
  - url: /assets/images/voronoi_monge.gif
    image_path: /assets/images/voronoi_monge.gif
    alt: "MongeNet mesh discretization by a point cloud"
    title: "MongeNet mesh discretization by a point cloud"
  - url: /assets/images/voronoi_unif.gif
    image_path: /assets/images/voronoi_unif.gif
    alt: "Standard random uniform mesh discretization by a point cloud"
    title: "Standard random uniform mesh discretization by a point cloud" 
gallery_airplane:
  - url: /assets/images/avion_mongenet.png
    image_path: /assets/images/avion_mongenet.png
    alt: "MongeNet mesh discretization by a point cloud"
    title: "MongeNet mesh discretization by a point cloud"
  - url: /assets/images/avion_uniform.png
    image_path: /assets/images/avion_uniform.png
    alt: "Standard random uniform mesh discretization by a point cloud"
    title: "Standard random uniform mesh discretization by a point cloud" 
---

Codes and datasets will be available soon!

## Abstract

Neural Rendering representations have significantly contributed to the field of 3D computer vision. Given their potential, considerable efforts have been invested to improve their performance. Nonetheless, the essential question of selecting the training view (changing for every rendered scene) is yet to be thoroughly investigated. This key aspect plays a vital role in achieving high-quality results and aligns with the well-known tenet of deep learning: **"garbage in, garbage out"**. In this paper, we first illustrate the importance of view selection by demonstrating how a basic rotation of the rendered object within the most widely used dataset can lead to consequential shifts in the performance rankings of state-of-the-art techniques. To address this challenge, we introduce a comprehensive framework to assess the impact of various training view selection methods and propose novel view selection methods. Significant improvements can be achieved without leveraging error or uncertainty estimation but focusing on uniform view coverage of the reconstructed object, resulting in a training-free approach. Using this technique, we show that similar performances can be achieved faster by using fewer views. We conduct extensive experiments on both synthetic datasets and realistic data to demonstrate the effectiveness of our proposed method compared with random, conventional error-based, and uncertainty-guided view selection.

## A Robust Testing Dataset

## Visual Comparisons of Different View Selection Methods

<br/>

If you find this work useful, please cite
```
@InProceedings{Xiao:CVPR24:NeRFDirector,
    author    = {Xiao, Wenhui and Santa Cruz, Rodrigo and Ahmedt-Aristizabal, David and Salvado, Olivier and Fookes, Clinton and Lebrat, Leo},
    title     = {NeRF Director: Revisiting View Selection in Neural Volume Rendering},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024}
}
```

## Acknowledgment 
This research was supported by [Maxwell plus](https://maxwellplus.com/)
