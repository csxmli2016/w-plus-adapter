## When StyleGAN Meets Stable Diffusion: a ${\mathcal{W}_+}$ Adapter for Personalized Image Generation

[Xiaoming Li](https://csxmli2016.github.io/), [Xinyu Hou](https://itsmag11.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University

<img src="./figures/fig1.png" width="800px">

<p align="justify">Text-to-image diffusion models have remarkably excelled in producing diverse, high-quality, and photo-realistic images. This advancement has spurred a growing interest in incorporating specific identities into generated content. Most current methods employ an inversion approach to embed a target visual concept into the text embedding space using a single reference image. However, the newly synthesized faces either closely resemble the reference image in terms of facial attributes, such as expression, or exhibit a reduced capacity for identity preservation. Text descriptions intended to guide the facial attributes of the synthesized face may fall short, owing to the intricate entanglement of identity information with identity-irrelevant facial attributes derived from the reference image. To address these issues, we present the novel use of the extended StyleGAN embedding space $\mathcal{W}_+$, to achieve enhanced identity preservation and disentanglement for diffusion models. By aligning this semantically meaningful human face latent space with text-to-image diffusion models, we succeed in maintaining high fidelity in identity preservation, coupled with the capacity for semantic editing. Additionally, we propose new training objectives to balance the influences of both prompt and identity conditions, ensuring that the identity-irrelevant background remains unaffected during facial attribute modifications. Extensive experiments reveal that our method adeptly generates personalized text-to-image outputs that are not only compatible with prompt descriptions but also amenable to common StyleGAN editing directions in diverse settings. </p>

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
<p align="justify">Leveraging our $\mathcal{W}_+$ adapter, our approach successfully generates images that are not only compatible with text descriptions but also more effectively retain the target identity. Additionally, our method allows for the editing of facial attributes along the $\delta w$ direction, causing only minor alterations in the non-facial regions (illustrated in the last column). Furthermore, our approach can be seamlessly adapted to other pre-trained SD models without the need for additional fine-tuning, while retaining its editing capabilities. This versatility is exemplified in the last row of the figure aside, which showcases our method's effectiveness with the dreamlike-anime model.</p>

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


