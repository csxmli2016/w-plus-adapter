
<div align="center">
<h1>When StyleGAN Meets Stable Diffusion:<br> a ${\mathcal{W}_+}$ Adapter for Personalized Image Generation</h1>

[Xiaoming Li](https://csxmli2016.github.io/), [Xinyu Hou](https://itsmag11.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

<div>
    <sup></sup>S-Lab, Nanyang Technological University
</div>

[Paper](#) | [Project Page](#)

<p><B>We propose a</B> $\mathcal{W}_+$ <B>adapter, a method that aligns the face latent space </B> $\mathcal{W}_+$ <B> of StyleGAN with text-to-image diffusion models, achieving high fidelity in identity preservation and semantic editing.</B></p>

<img src="./figures/fig1.png" width="800px">

<p align="justify">Given a single reference image (thumbnail in the top left), our $\mathcal{W}_+$ adapter not only integrates the identity into the text-to-image generation accurately but also enables modifications of facial attributes along the $\Delta w$ trajectory derived from StyleGAN. The text prompt is ``a woman wearing a spacesuit in a forest''.  </p>

</div>

## TODO
- [] Release the source code and model.
- [] Extend to more diffusion models.


## The Framework of $\mathcal{W}_+$ Adapter

<img src="./figures/pipeline.png" width="800px">

<p align="justify">Our approach is capable of generating images that preserve identity while allowing for semantic edits, requiring just a single reference image for inference. This capability is realized by innovatively aligning StyleGAN's $\mathcal{W}_+$ latent space with the diffusion model. The training of our $\mathcal{W}_+$ adapter is divided into two stages. In Stage I, we establish a mapping from $\mathcal{W}_+$ to SD latent space, using the resulting projection as an additional identity condition to synthesize center-aligned facial images of a specified identity. In Stage II, this personalized generation process is expanded to accommodate more dynamic, "in-the-wild" settings, ensuring adaptability to a variety of textual prompts.</p>

## Comparison of Face Attributes Editing Using Ours (Stage I) and e4e

<img src="./figures/aba_stage1.png" width="800px">

## $w_+$ Interpolation
<img src="./figures/aba_w_interpolation.png" width="800px">
<p align="justify">Visual results of w+ embeddings interpolation from two real-world references. The prompts are “one person wearing suit and tie in a garden” and “one person wearing a blue shirt by a secluded waterfall”, respectively</p>


## Visual Comparison

<img src="./figures/vis_compare.png" width="800px">
<p align="justify">Leveraging our $\mathcal{W}_+$ adapter, our approach successfully generates images that are not only compatible with text descriptions but also more effectively retain the target identity. Additionally, our method allows for the editing of facial attributes along the $\Delta w$ direction, causing only minor alterations in the non-facial regions (illustrated in the last column). Furthermore, our approach can be seamlessly adapted to other pre-trained SD models without the need for additional fine-tuning, while retaining its editing capabilities. This versatility is exemplified in the last row of the figure aside, which showcases our method's effectiveness with the dreamlike-anime model.</p>

## License
This project is licensed under <a rel="license" href="https://github.com/csxmli2016/w-plus-adapter/blob/main/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.

## Acknowledgement
This project is built based on the excellent [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) and [FreeU](https://github.com/ChenyangSi/FreeU).


## Citation

```
@article{li2023w-plus-adapter,
author = {Li, Xiaoming and Hou, Xinyu and Loy, Chen Change},
title = {When StyleGAN Meets Stable Diffusion: a $\mathcal{W}_+$ Adapter for Personalized Image Generation},
journal = {arXiv preprint arXiv},
year = {2023}
}
```


