
<div align="center">
<h1>When StyleGAN Meets Stable Diffusion:<br> a ${\mathcal{W}_+}$ Adapter for Personalized Image Generation</h1>

[Xiaoming Li](https://csxmli2016.github.io/), [Xinyu Hou](https://itsmag11.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

<div>
    <sup></sup>S-Lab, Nanyang Technological University
</div>

[Paper](#) | [Project Page](https://csxmli2016.github.io/projects/w-plus-adapter/)

<p><B>We propose a</B> $\mathcal{W}_+$ <B>adapter, a method that aligns the face latent space </B> $\mathcal{W}_+$ <B> of StyleGAN with text-to-image diffusion models, achieving high fidelity in identity preservation and semantic editing.</B></p>

<img src="./figures/fig1.png" width="800px">

<p align="justify">Given a single reference image (thumbnail in the top left), our $\mathcal{W}_+$ adapter not only integrates the identity into the text-to-image generation accurately but also enables modifications of facial attributes along the $\Delta w$ trajectory derived from StyleGAN. The text prompt is ``a woman wearing a spacesuit in a forest''.  </p>

</div>

## TODO
- [] Release the source code and model.
- [] Extend to more diffusion models.


## License
This project is licensed under <a rel="license" href="https://github.com/csxmli2016/w-plus-adapter/blob/main/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.

## Acknowledgement
This project is built based on the excellent [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter).


## Citation

```
@article{li2023w-plus-adapter,
author = {Li, Xiaoming and Hou, Xinyu and Loy, Chen Change},
title = {When StyleGAN Meets Stable Diffusion: a $\mathcal{W}_+$ Adapter for Personalized Image Generation},
journal = {arXiv preprint arXiv},
year = {2023}
}
```


