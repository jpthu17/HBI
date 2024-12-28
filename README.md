<div align="center">
  
# 【CVPR'2023 Highlight🔥&TPAMI】Video-Text as Game Players: Hierarchical Banzhaf Interaction for Cross-Modal Representation Learning
  
[![Conference](http://img.shields.io/badge/CVPR-2023(Highlight)-FFD93D.svg)](https://cvpr.thecvf.com/)
[![Project](http://img.shields.io/badge/Project-HBI-4D96FF.svg)](https://jpthu17.github.io/HBI/)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2303.14369-FF6B6B.svg)](https://arxiv.org/abs/2303.14369)
</div>

The implementation of CVPR 2023 Highlight (Top 10%) paper [Video-Text as Game Players: Hierarchical Banzhaf Interaction for Cross-Modal Representation Learning](https://arxiv.org/abs/2303.14369).

In this paper, we creatively model video-text as game players with multivariate cooperative game theory to wisely handle the uncertainty during fine-grained semantic interaction with diverse granularity, flexible combination, and vague intensity.

## 📌 Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@article{jin2024hierarchical,
  title={Hierarchical Banzhaf Interaction for General Video-Language Representation Learning},
  author={Jin, Peng and Li, Hao and Yuan, Li and Yan, Shuicheng and Chen, Jie},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}

@inproceedings{jin2023video,
  title={Video-text as game players: Hierarchical banzhaf interaction for cross-modal representation learning},
  author={Jin, Peng and Huang, Jinfa and Xiong, Pengfei and Tian, Shangxuan and Liu, Chang and Ji, Xiangyang and Yuan, Li and Chen, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2472--2482},
  year={2023}
}
```

<details open><summary>💡 I also have other text-video retrieval projects that may interest you ✨. </summary><p>

> [**DiffusionRet: Generative Text-Video Retrieval with Diffusion Model**](https://arxiv.org/abs/2303.09867)<br>
> Accepted by ICCV 2023 | [[DiffusionRet Code]](https://github.com/jpthu17/DiffusionRet)<br>
> Peng Jin, Hao Li, Zesen Cheng, Kehan Li, Xiangyang Ji, Chang Liu, Li Yuan, Jie Chen

> [**Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations**](https://arxiv.org/abs/2211.11427)<br>
> Accepted by NeurIPS 2022 | [[EMCL Code]](https://github.com/jpthu17/EMCL)<br>
> Peng Jin, Jinfa Huang, Fenglin Liu, Xian Wu, Shen Ge, Guoli Song, David Clifton, Jie Chen

> [**Text-Video Retrieval with Disentangled Conceptualization and Set-to-Set Alignment**](https://arxiv.org/abs/2305.12218)<br>
> Accepted by IJCAI 2023 | [[DiCoSA Code]](https://github.com/jpthu17/DiCoSA)<br>
> Peng Jin, Hao Li, Zesen Cheng, Jinfa Huang, Zhennan Wang, Li Yuan, Chang Liu, Jie Chen
</p></details>

## 📣 Updates
* **[2023/10/15]**: We release our [pre-trained estimator weights](https://github.com/jpthu17/HBI#train-the-banzhaf-interaction-estimator). If you want to apply a to other tasks, you can initialize a new estimator with the weights we provide. If you want better performance, you can train the estimator with a smaller learning rate and more epochs.
* **[2023/10/11]**: We release code for Banzhaf Interaction estimator. Recommended running parameters will be provided shortly, and we will also release our pre-trained estimator weights.
* **[2023/10/08]**: I am working on the code for Banzhaf Interaction estimator, which is expected to be released soon.
* **[2023/06/28]**: Release code for reimplementing the experiments in the paper.
* **[2023/03/28]**: Our **HBI** has been selected as a Highlight paper at CVPR 2023! (Top 2.5% of 9155 submissions).
* **[2023/02/28]**: We will release the code asap. (I am busy with other DDLs. After that, I will open the source code as soon as possible. Please understand.)

  
## ⚡ Demo
<div align="center">
  
https://user-images.githubusercontent.com/53246557/221760113-4a523e7e-d743-4dff-9f16-357ab0be0d5b.mp4
</div>


## 😍 Visualization

### Example 1
<div align=center>
<img src="static/images/Visualization_1.png" width="800px">
</div>

<details>
<summary><b>More examples</b></summary>
  
### Example 2
<div align=center>
<img src="static/images/Visualization_2.png" width="800px">
</div>

### Example 3
<div align=center>
<img src="static/images/Visualization_3.png" width="800px">
</div>

### Example 4
<div align=center>
<img src="static/images/Visualization_4.png" width="800px">
</div>

### Example 5
<div align=center>
<img src="static/images/Visualization_5.png" width="800px">
</div>

### Example 6
<div align=center>
<img src="static/images/Visualization_6.png" width="800px">
</div>

### Example 7
<div align=center>
<img src="static/images/Visualization_0.png" width="800px">
</div>

</details>

## 🚀 Quick Start
### Setup

#### Setup code environment
```shell
conda create -n HBI python=3.9
conda activate HBI
pip install -r requirements.txt
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Download CLIP Model
```shell
cd HBI/models
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
# wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
# wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

#### Download Datasets
<div align=center>

|Datasets|Google Cloud|Baidu Yun|Peking University Yun|
|:--------:|:--------------:|:-----------:|:-----------:|
| MSR-VTT | [Download](https://drive.google.com/drive/folders/1LYVUCPRxpKMRjCSfB_Gz-ugQa88FqDu_?usp=sharing) | [Download](https://pan.baidu.com/s/1Gdf6ivybZkpua5z1HsCWRA?pwd=enav) | [Download](https://disk.pku.edu.cn/link/AA6A028EE7EF5C48A788118B82D6ABE0C5) |
| MSVD | [Download](https://drive.google.com/drive/folders/18EXLWvCCQMRBd7-n6uznBUHdP4uC6Q15?usp=sharing) | [Download](https://pan.baidu.com/s/1hApFdxgV3TV2TCcnM_yBiA?pwd=kbfi) | [Download](https://disk.pku.edu.cn/link/AA6BD6FC1A490F4D0E9C384EF347F0D07F) |
| ActivityNet | TODO | [Download](https://pan.baidu.com/s/1tI441VGvN3In7pcvss0grg?pwd=2ddy) | [Download](https://disk.pku.edu.cn/link/AAE744E6488E2049BD9412738E14AAA8EA) |
| DiDeMo | TODO | [Download](https://pan.baidu.com/s/1Tsy9nb1hWzeXaZ4xr7qoTg?pwd=c842) | [Download](https://disk.pku.edu.cn/link/AA14E48D1333114022B736291D60350FA5) |

</div>

#### Train the Banzhaf Interaction Estimator

Train the estimator according to the label generated by the BanzhafInteraction in HBI/models/banzhaf.py. 

The training code is provided in banzhaf_estimator.py. We provide our trained weights, and if you want to apply a to other tasks, you can initialize a new estimator with the weights we provide.

We have tested the performance of [Estimator_1e-2_epoch6](https://drive.google.com/file/d/1GYDUIlEA1Fe9E_9IhE4Thgm5mo2ZcRa6/view?usp=sharing) with R@1 of 48.2 ([log](https://drive.google.com/file/d/1F-QvhvFj9s7tqoLnVwuUKCIbnLr2MHBq/view?usp=sharing)) on the MSR-VTT dataset. If you want better performance, you can train the estimator with a smaller learning rate and more epochs.

<div align=center>

|   Models    | Google Cloud | Baidu Yun |Peking University Yun| log|
|:-----------:|:------------:|:---------:|:-----------:|:-----------:|
|   Estimator_1e-2_epoch1   |     [Download](https://drive.google.com/file/d/1U2QsawOhBaPthZd13_pi_Qhi6kgvT1GB/view?usp=sharing)     |  [Download](https://pan.baidu.com/s/1mxpSHAxEH8qz59ROJTwH7A?pwd=ewsp)     | [Download](https://disk.pku.edu.cn:443/link/3E245D48A388A9DDCA9B8A45BE31C594)  | [log](https://drive.google.com/file/d/1rD1ywMgP_q_M-Njz7QVC0mOX0mM4wbUH/view?usp=sharing)  |
|   Estimator_1e-2_epoch2   |     [Download](https://drive.google.com/file/d/1cdv6058pu2xhroI4gk4gl60IT7wWIDkj/view?usp=sharing)     |  [Download](https://pan.baidu.com/s/1Yo-fve2Oq1_KoLKQwztD5w?pwd=3mlo)      | [Download](https://disk.pku.edu.cn:443/link/AE8F75FC2A97DD903C4D562D965B6728)  | [log](https://drive.google.com/file/d/1rD1ywMgP_q_M-Njz7QVC0mOX0mM4wbUH/view?usp=sharing)  |
|   Estimator_1e-2_epoch3   |     [Download](https://drive.google.com/file/d/1XjTWpyRFy0SmzsbyZ2YS2UczEEEgMppP/view?usp=sharing)     |  [Download](https://pan.baidu.com/s/1FPFlOtAVU27KCFH9i4eWZg?pwd=p5qf)      | [Download](https://disk.pku.edu.cn:443/link/0ACDF14C9CA901898F15B4CC4F8C0E30)  | [log](https://drive.google.com/file/d/1rD1ywMgP_q_M-Njz7QVC0mOX0mM4wbUH/view?usp=sharing)  |
|   Estimator_1e-2_epoch4   |     [Download](https://drive.google.com/file/d/12b6Pjg5HrIRhMqq5KkLF_FKXY4RHv4Hn/view?usp=sharing)     |  [Download](https://pan.baidu.com/s/1LP99MFizCr_bgt9DtlLweg?pwd=skn3)     | [Download](https://disk.pku.edu.cn:443/link/615B6ABAB30E5A3064310ACAC28BC5CD)  | [log](https://drive.google.com/file/d/1rD1ywMgP_q_M-Njz7QVC0mOX0mM4wbUH/view?usp=sharing)  |
|   Estimator_1e-2_epoch5   |     [Download](https://drive.google.com/file/d/1oLil8xQ0JwI2QWGNj8ghs_x1nI-mHigp/view?usp=sharing)     |  [Download](https://pan.baidu.com/s/1ORJkUmLe2fhMySTQrlKWcw?pwd=c8w8)      | [Download](https://disk.pku.edu.cn:443/link/5E1DEA84D402AFFFB304F571949736B1)  | [log](https://drive.google.com/file/d/1rD1ywMgP_q_M-Njz7QVC0mOX0mM4wbUH/view?usp=sharing)  |
|   Estimator_1e-2_epoch6   |     [Download](https://drive.google.com/file/d/1GYDUIlEA1Fe9E_9IhE4Thgm5mo2ZcRa6/view?usp=sharing)    |  [Download](https://pan.baidu.com/s/1Kmn3laMFrG8WWQqNIyK69Q?pwd=79eb)     | [Download](https://disk.pku.edu.cn:443/link/7893AD6A50BAFCA342456B0B04C99419)  | [log](https://drive.google.com/file/d/1rD1ywMgP_q_M-Njz7QVC0mOX0mM4wbUH/view?usp=sharing)  |

</div>

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=4 \
banzhaf_estimator.py \
--do_train 1 \
--workers 8 \
--n_display 1 \
--epochs 10 \
--lr 1e-2 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 24 \
--max_frames 12 \
--video_framerate 1 \
--output_dir ${OUTPUT_PATH} 
```

### Text-video Retrieval
<div align=center>

|Checkpoint|Google Cloud|Baidu Yun|Peking University Yun|
|:--------:|:--------------:|:-----------:|:-----------:|
| MSR-VTT | [Download](https://drive.google.com/file/d/1hoV9vsT0-KIjjIRPIB9D4dMXwrckvSLk/view?usp=sharing) | [Download](https://pan.baidu.com/s/1WWlpoSAUII3KH6KNsq7VSQ?pwd=pkph) | [Download](https://disk.pku.edu.cn:443/link/424DFFAC5D2CB600E73BCB67C05A73FD) |
| ActivityNet | [Download](https://drive.google.com/file/d/1TRUAl17Wj2g2cyxWC5HUPflUo7eg78uu/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1ynAaE0NWXx0LHhUZCC0uww?pwd=ta8v) | [Download](https://disk.pku.edu.cn:443/link/A7BDBF989B3E2C6356283ED01FBAACF2) |

</div>

#### Eval on MSR-VTT
```shell
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=2 \
main_retrieval.py \
--do_eval 1 \
--workers 8 \
--n_display 50 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 24 \
--max_frames 12 \
--video_framerate 1 \
--init_model ${CHECKPOINT_PATH} \
--output_dir ${OUTPUT_PATH} 
```

#### Train on MSR-VTT
```shell
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=2 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 24 \
--max_frames 12 \
--video_framerate 1 \
--estimator ${ESTIMATOR_PATH} \
--output_dir ${OUTPUT_PATH} \
--kl 2 \
--skl 1
```

#### Eval on ActivityNet Captions
```shell
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=2 \
main_retrieval.py \
--do_eval 1 \
--workers 8 \
--n_display 50 \
--batch_size_val 128 \
--anno_path ${DATA_PATH}/ActivityNet \
--video_path ${DATA_PATH}/ActivityNet/Activity_Videos \
--datatype activity \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--init_model ${CHECKPOINT_PATH} \
--output_dir ${OUTPUT_PATH} 
```

#### Train on ActivityNet Captions
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=8 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 10 \
--epochs 10 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${DATA_PATH}/ActivityNet \
--video_path ${DATA_PATH}/ActivityNet/Activity_Videos \
--datatype activity \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--estimator ${ESTIMATOR_PATH} \
--output_dir ${OUTPUT_PATH} \
--kl 2 \
--skl 1
```

### Video-question Answering
<div align=center>

|Checkpoint|Google Cloud|Baidu Yun|Peking University Yun|
|:--------:|:--------------:|:-----------:|:-----------:|
| MSR-VTT-QA | [Download](https://drive.google.com/file/d/15GZXMaPvowL4GgxtB9ETvb8vivdcE8Wd/view?usp=sharing) | [Download](https://pan.baidu.com/s/1a959PS2EaYHxcYyrrQ4odQ?pwd=r34t) | [Download](https://disk.pku.edu.cn:443/link/DE99ECAD7C1E7F550A2753B561086CDF) |

</div>

#### Eval on MSR-VTT-QA

```shell
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=2 \
main_vqa.py \
--do_eval \ 
--num_thread_reader=8 \
--train_csv data/MSR-VTT/qa/train.jsonl \
--val_csv data/MSR-VTT/qa/test.jsonl \
--data_path data/MSR-VTT/qa/train_ans2label.json \
--features_path ${DATA_PATH}/MSRVTT_Videos \
--max_words 32 \
--max_frames 12 \
--batch_size_val 16 \
--datatype msrvtt \
--expand_msrvtt_sentences  \
--feature_framerate 1 \
--freeze_layer_num 0  \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d \
--init_model ${CHECKPOINT_PATH} \
--output_dir ${OUTPUT_PATH}
```

#### Train on MSR-VTT-QA

```shell
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=2 \
main_vqa.py \
--do_train \ 
--num_thread_reader=8 \
--epochs=5 \
--batch_size=32 \
--n_display=50 \
--train_csv data/MSR-VTT/qa/train.jsonl \
--val_csv data/MSR-VTT/qa/test.jsonl \
--data_path data/MSR-VTT/qa/train_ans2label.json \
--features_path ${DATA_PATH}/MSRVTT_Videos \
--lr 1e-4 \
--max_words 32 \
--max_frames 12 \
--batch_size_val 16 \
--datatype msrvtt \
--expand_msrvtt_sentences  \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0  \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d \
--estimator ${ESTIMATOR_PATH} \
--output_dir ${OUTPUT_PATH} \
--kl 2 \
--skl 1
```

## 🎗️ Acknowledgments
Our code is based on [EMCL](https://github.com/jpthu17/EMCL), [CLIP](https://github.com/openai/CLIP), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/) and [DRL](https://github.com/foolwood/DRL). We sincerely appreciate for their contributions.
