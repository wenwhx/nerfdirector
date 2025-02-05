---
layout: single
author_profile: True
classes: wide
excerpt: "Revisiting View Selection in Neural Volume Rendering<br/>CVPR 2024"
header:
  overlay_image: /assets/images/banner_ext.png
  overlay_filter: 0.5
  caption: "
  "
  actions:
    - label: "Paper(ArXiv)"
      url: "https://arxiv.org/abs/2406.08839"
    - label: "Code"
      url: "https://github.com/wenwhx/nerfdirector"
    - label: "Datasets"
      url: "https://data.csiro.au/collection/csiro:63796"
gallery_default_test:
  - url: /assets/images/default.gif
    image_path: /assets/images/default.gif
    alt: "Default pose of the original testing camera and 10 additional rotated versions"
    title: "Default pose of the original testing camera and 10 additional rotated versions"
  - url: /assets/images/original_rank.png
    image_path: /assets/images/original_rank.png
gallery_propose_test:
  - url: /assets/images/new.gif
    image_path: /assets/images/new.gif
    alt: "Default pose of our proposed testing camera and 10 additional rotated versions"
    title: "Default pose of our proposed testing camera and 10 additional rotated versions"
  - url: /assets/images/propose_rank.png
    image_path: /assets/images/propose_rank.png
---


Our code and datasets are available now!

## Abstract

Neural Rendering representations have significantly contributed to the field of 3D computer vision. Given their potential, considerable efforts have been invested to improve their performance. Nonetheless, the essential question of selecting the training view (changing for every rendered scene) is yet to be thoroughly investigated. This key aspect plays a vital role in achieving high-quality results and aligns with the well-known tenet of deep learning: **"garbage in, garbage out"**. In this paper, we first illustrate the importance of view selection by demonstrating how a basic rotation of the rendered object within the most widely used dataset can lead to consequential shifts in the performance rankings of state-of-the-art techniques. To address this challenge, we introduce a comprehensive framework to assess the impact of various training view selection methods and propose novel view selection methods. Significant improvements can be achieved without leveraging error or uncertainty estimation but focusing on uniform view coverage of the reconstructed object, resulting in a training-free approach. Using this technique, we show that similar performances can be achieved faster by using fewer views. We conduct extensive experiments on both synthetic datasets and realistic data to demonstrate the effectiveness of our proposed method compared with random, conventional error-based, and uncertainty-guided view selection.

## A Robust Testing Set

<figure>
  <figcaption>360 visual comparisons between default testing camera distribution of the NeRF Synthetic dataset and our proposed one. <em>Left:</em> Default testing cameras are in a constant simple track. <em>Right:</em> Proposed testing cameras are distributed evenly around the target.</figcaption>
  <div>
  <video id="blender" width="48%" autoplay loop muted controls>
    <source src="/nerfdirector/assets/images/out_classic.mp4" type="video/mp4">
  </video>

  <video id="blender" width="48%" autoplay loop muted controls>
    <source src="/nerfdirector/assets/images/out_new.mp4" type="video/mp4">
  </video>
  </div>
</figure>

### Ranking inversion of SOTA NeRF models
We apply 10 rotations to the default testing cameras and generate 10 additional testing sets. We evaluate the pre-trained checkpoint of 4 SOTA NeRF models (JaxNeRF, MipNeRF, Plenoxels, and InstantNGP) on all these testing sets. The ranking are variated across different rotations.
{% include gallery id="gallery_default_test" caption="Ranking inversion across different rotations occurs on the original testing camera."%}

Our proposed test set, evenly distributed on the sphere with the target as the center, can provide a consistent comparison across different rotations.
{% include gallery id="gallery_propose_test" caption="Our proposed test sest provides a robust evaluation."%}

### Proposed test set on TanksAndTemple dataset
<figure>
  <figcaption>360 visual comparisons of view coverage between default testing camera distribution of the TanksAndTemple dataset and our proposed one. <em>Left:</em> Default test set only coverages parts of the target. <em>Right:</em> Proposed testing cameras achieves a more uniform coverage of the target.</figcaption>
  <div>
  <video id="tnt" width="48%" autoplay loop muted controls>
    <source src="/nerfdirector/assets/images/m60_test.webm" type="video/webm">
  </video>

  <video id="tnt" width="48%" autoplay loop muted controls>
    <source src="/nerfdirector/assets/images/m60_ours.webm" type="video/webm">
  </video>
  </div>
</figure>

## Visual Comparisons of Different View Selection Methods

<figure>
  <div>
  <video id="v0" width="100%" autoplay loop muted controls>
    <source src="/nerfdirector/assets/images/truck_old.webm" type="video/webm">
  </video>

  <video id="v1" width="100%" autoplay loop muted controls>
    <source src="/nerfdirector/assets/images/truck_new.webm" type="video/webm">
  </video>
  </div>
  <figcaption>Visualization of 80 camera selected via different view selection methods on 'Truck' scene. <em>Top:</em> Ramdom Sampling (RS) on default. <em>Bottom:</em> Proposed Farthest View Sampling (FVS) distributing cameras evenly around the target object.</figcaption>
</figure>

<canvas id="test" width="400" height="400"></canvas>

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

