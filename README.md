# Patch-wise Contrastive Style Learning for Instagram Filter Removal

![][results]

[arXiv][arxiv] | [Demo][demo]

> **Patch-wise Contrastive Style Learning for Instagram Filter Removal**<br>
> Furkan Kınlı, Barış Özcan, Furkan Kıraç <br>
> *Accepted to  NTIRE2022: New Trends in Image Restoration and Enhancement workshop and challenges on image and video processing in conjunction with CVPR 2022.* <br>
>
>**Abstract:** Image-level corruptions and perturbations degrade the performance of CNNs on different downstream vision tasks. Social media filters are one of the most common resources of various corruptions and perturbations for real-world visual analysis applications. The negative effects of these distractive factors can be alleviated by recovering the original images with their pure style for the inference of the downstream vision tasks. Assuming these filters substantially inject a piece of additional style information to the social media images, we can formulate the problem of recovering the original versions as a reverse style transfer problem. We introduce Contrastive Instagram Filter Removal Network (CIFR), which enhances this idea for Instagram filter removal by employing a novel multi-layer patch-wise contrastive style learning mechanism. Experiments show our proposed strategy produces better qualitative and quantitative results than the previous studies. Moreover, we present the results of our additional experiments for proposed architecture within different settings. Finally, we present the inference outputs and quantitative comparison of filtered and recovered images on localization and segmentation tasks to encourage the main motivation for this problem.

## Description
The official implementation of the paper titled "Patch-wise Contrastive Style Learning for Instagram Filter Removal".
We propose a patch-wise multi-layer contrastive strategy for removing Instagram filters from the images by assuming the affects of filters as the style information.


## Requirements
To install requirements:

```
pip install -r requirements.txt
```

## Architecture
![][model]

## Dataset
[**IFFI dataset**][dataset]
contains 600 images and with their 16 different filtered versions for each. In particular, mostly-used
16 filters as follows: *1977*, *Amaro*, *Brannan*, *Clarendon*, *Gingham*,
*He-Fe*, *Hudson*, *Lo-Fi*, *Mayfair*, *Nashville*, *Perpetua*, *Sutro*,
*Toaster*, *Valencia*, *Willow*, *X-Pro II*.

## Training

To train CIFR from the scratch in the paper, run this command:

```
python main.py --base_cfg config.yml --dataset IFFI --dataset_dir /path/to/dataset
```

## Evaluation

To evaluate CIFR on IFFI dataset, run:

```
python main.py --base_cfg config.yaml -t -w ifrnet.pth --dataset IFFI --dataset_dir /path/to/dataset
```

[comment]: <> (## Citation)

[comment]: <> (```)

[comment]: <> (@misc{kınlı2021instagram,)

[comment]: <> (      title={Instagram Filter Removal on Fashionable Images}, )

[comment]: <> (      author={Furkan Kınlı and Barış Özcan and Furkan Kıraç},)

[comment]: <> (      year={2021},)

[comment]: <> (      eprint={2104.05072},)

[comment]: <> (      archivePrefix={arXiv},)

[comment]: <> (      primaryClass={cs.CV})

[comment]: <> (})

[comment]: <> (```)

## Contacts
Please feel free to open an issue or to send an e-mail to ```furkan.kinli@ozyegin.edu.tr```

[results]: images/paper/results.png
[model]: images/paper/arch.png
[dataset]: https://github.com/birdortyedi/instagram-filter-removal-pytorch
[demo]: https://huggingface.co/spaces/birdortyedi/cifr-pytorch
[arxiv]: https://arxiv.org/abs/2204.07486