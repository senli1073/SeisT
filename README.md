

[![TGRS](https://img.shields.io/badge/IEEE_TGRS_(2024)-5908215-blue)](https://doi.org/10.1109/TGRS.2024.3371503)
[![arXiv](https://img.shields.io/badge/arXiv-2310.01037-b31b1b)](https://arxiv.org/abs/2310.01037)
![License](https://img.shields.io/github/license/senli1073/SeisT)
![LastCommit](https://img.shields.io/github/last-commit/senli1073/SeisT)
------------------

- [SeisT Architecture](#seist-architecture)
- [Introduction](#introduction)
- [Usage](#usage)
  - [Data preparation](#data-preparation)
  - [Training](#training)
  - [Fine-tuning](#fine-tuning)
  - [Testing](#testing)
- [Citation](#citation)
- [Reporting Bugs](#reporting-bugs)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## SeisT Architecture
<p align="center">
  <img src="https://raw.githubusercontent.com/senli1073/SeisT/main/images/SeisT_Architecture.png">
</p>

## Introduction
SeisT is a backbone for seismic signal processing, which can be used for multiple seismic monitoring tasks such as earthquake detection, seismic phase picking, first-motion polarity classification, magnitude estimation, back-azimuth estimation, and epicentral distance estimation.

This repository also provides some baseline models implemented by Pytorch under `./models`, such as PhaseNet, EQTransformer, DitingMotion, MagNet, BAZ-Network, and distPT-Network. 

NOTE: The model weights included in this repository serve as the basis for performance evaluation in the paper. They have been evaluated using identical training/testing data and a consistent training strategy, thereby affirming the effectiveness of SeisT. Nevertheless, if you intend to employ these models in practical engineering applications, it is crucial to retrain the SeisT models with larger datasets to align with the specific demands of engineering applications.

## Usage

### Data Preparation

- **For training and evaluation**
  
  Create a new file named `yourdata.py` in the directory `dataset/` to read the metadata and seismograms of the dataset. Then the `@register_dataset` decorator needs to be used to register your dataset. 

  (Please refer to the code samples, such as `datasets/DiTing.py` and `datasets/PNW.py`)

- **For model deployment**

  Follow the steps in `demo_predict.py` and rewrite the `load_data` function.

### Training

- **Model**<br/>
  Before starting training, please make sure that your model file is in the directory `models/` and is registered by using `@register_model`. The  available models in the project can be inspected using the following method: 
  ```Python
  >>> from models import get_model_list
  >>> get_model_list()
  ['eqtransformer', 'phasenet', 'magnet', 'baz_network', 'distpt_network', 'ditingmotion', 'seist_s_dpk', 'seist_m_dpk', 'seist_l_dpk', 'seist_s_pmp', 'seist_m_pmp', 'seist_l_pmp', 'seist_s_emg', 'seist_m_emg', 'seist_l_emg', 'seist_s_baz', 'seist_m_baz', 'seist_l_baz', 'seist_s_dis', 'seist_m_dis', 'seist_l_dis']
  ```
  The task names and their abbreviations in this project are shown in the table below:

  <table><tbody>

  <th valign="bottom">Task</th>
  <th valign="bottom">Abbreviation</th>

  <tr><td align="left">Detection & Phase Picking</td>
  <td align="left">dpk</td>

  <tr><td align="left">First-Motion Polarity Classification</td>
  <td align="left">pmp</td>

  <tr><td align="left">Back-Azimuth Estimation</td>
  <td align="left">baz</td>

  <tr><td align="left">Magnitude Estimation</td>
  <td align="left">emg</td>

  <tr><td align="left">Epicentral Distance Estimation</td>
  <td align="left">dis</td>

  </tbody></table>

- **Model Configuration**<br/>
  The configurations of the loss functions, labels, and the corresponding models are in `config.py` which also provides a detailed explanation of all the fields.


- **Start training**<br/>
  To start training with a CPU or a single GPU, please use the following command to start training:
  ```Shell
  python main.py \
    --seed 0 \
    --mode "train_test" \
    --model-name "seist_m_dpk" \
    --log-base "./logs" \
    --device "cuda:0" \
    --data "/root/data/Datasets/Diting50hz" \
    --dataset-name "diting" \
    --data-split true \
    --train-size 0.8 \
    --val-size 0.1 \
    --shuffle true \
    --workers 8 \
    --in-samples 8192 \
    --augmentation true \
    --epochs 200 \
    --patience 30 \
    --batch-size 500
  ```
  
  To start training with multiple GPUs, please use `torchrun` to start training:
  ```Shell
  torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    main.py \
      --seed 0 \
      --mode "train_test" \
      --model-name "seist_m_dpk" \
      --log-base "./logs" \
      --data "/root/data/Datasets/Diting50hz" \
      --dataset-name "diting" \
      --data-split true \
      --train-size 0.8 \
      --val-size 0.1 \
      --shuffle true \
      --workers 8 \
      --in-samples 8192 \
      --augmentation true \
      --epochs 200 \
      --patience 30 \
      --batch-size 500
  ```
  
  There are also a variety of other custom arguments which are not mentioned above. Use the command `python main.py --help` to see more details.

  
### Fine-tuning

The following table provides the pre-trained checkpoints used in the paper:
<table><tbody>

<th valign="bottom">Task</th>
<th valign="bottom">Train set</th>
<th valign="bottom">SeisT-S</th>
<th valign="bottom">SeisT-M</th>
<th valign="bottom">SeisT-L</th>


<tr><td align="left">Detection & Phase Picking</td>
<td align="left">DiTing</td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_s_dpk_diting.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_m_dpk_diting.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_l_dpk_diting.pth">download</a></td>

<tr><td align="left">First-Motion Polarity Classification</td>
<td align="left">DiTing</td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_s_pmp_diting.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_m_pmp_diting.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_l_pmp_diting.pth">download</a></td>

<tr><td align="left">Back-Azimuth Estimation</td>
<td align="left">DiTing</td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_s_baz_diting.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_m_baz_diting.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_l_baz_diting.pth">download</a></td>

<tr><td align="left">Magnitude Estimation</td>
<td align="left">DiTing</td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_s_emg_diting.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_m_emg_diting.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_l_emg_diting.pth">download</a></td>

<tr><td align="left">Magnitude Estimation</td>
<td align="left">PNW</td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_s_emg_pnw.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_m_emg_pnw.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_l_emg_pnw.pth">download</a></td>

<tr><td align="left">Epicentral Distance Estimation</td>
<td align="left">DiTing</td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_s_dis_diting.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_m_dis_diting.pth">download</a></td>
<td align="center"><a href="https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_l_dis_diting.pth">download</a></td>

</tbody></table>

Use the "--checkpoint" argument to pass in the path of the pre-training weights.

### Testing
  To start training with a CPU or a single GPU, please use the following command to start testing:

  ```Shell
  python main.py \
    --seed 0 \
    --mode "test" \
    --model-name "seist_m_dpk" \
    --log-base "./logs" \
    --device "cuda:0" \
    --data "/root/data/Datasets/Diting50hz" \
    --dataset-name "diting" \
    --data-split true \
    --train-size 0.8 \
    --val-size 0.1 \
    --workers 8 \
    --in-samples 8192 \
    --batch-size 500
  ```
  
  To start training with multiple GPUs, please use `torchrun` to start testing:
  ```Shell
  torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    main.py \
      --seed 0 \
      --mode "test" \
      --model-name "seist_m_dpk" \
      --log-base "./logs" \
      --data "/root/data/Datasets/Diting50hz" \
      --dataset-name "diting" \
      --data-split true \
      --train-size 0.8 \
      --val-size 0.1 \
      --workers 8 \
      --in-samples 8192 \
      --batch-size 500
  ```

  It should be noted that the `train_size`, `val_size`, and `seed` in the test phase must be consistent with that training phase. Otherwise, the test results may be distorted.

## Citation

Paper: https://doi.org/10.1109/TGRS.2024.3371503

If you find this repo useful in your research, please consider citing:

```
@ARTICLE{10453976,
  author={Li, Sen and Yang, Xu and Cao, Anye and Wang, Changbin and Liu, Yaoqi and Liu, Yapeng and Niu, Qiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SeisT: A Foundational Deep-Learning Model for Earthquake Monitoring Tasks}, 
  year={2024},
  volume={62},
  pages={1-15},
  doi={10.1109/TGRS.2024.3371503}
}
```

The baseline models used in this paper:

- **PhaseNet**<br/>
  *Zhu, W., & Beroza, G. C. (2019). PhaseNet: A deep-neural-network-based seismic arrival-time picking method. Geophysical Journal International, 216(1), 261-273.*

- **EQTransformer**<br/>
  *Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., & Beroza, G. C. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature communications, 11(1), 3952.*

- **DiTingMotion**<br/>
  *Zhao, M., Xiao, Z., Zhang, M., Yang, Y., Tang, L., & Chen, S. (2023). DiTingMotion: A deep-learning first-motion-polarity classifier and its application to focal mechanism inversion. Frontiers in Earth Science, 11, 1103914.*

- **MagNet**<br/>
  *Mousavi, S. M., & Beroza, G. C. (2020). A machine‐learning approach for earthquake magnitude estimation. Geophysical Research Letters, 47(1), e2019GL085976.*

- **BAZ-Network** <br/>
  *Mousavi, S. M., & Beroza, G. C. (2020). Bayesian-Deep-Learning Estimation of Earthquake Location From Single-Station Observations. IEEE Transactions on Geoscience and Remote Sensing, 58(11), 8211-8224.*


## Reporting Bugs
Report bugs at https://github.com/senli1073/SeisT/issues.

If you are reporting a bug, please include:

- Operating system version.
- Versions of Python and libraries such as Pytorch.
- Steps to reproduce the bug.


## Acknowledgement
This project refers to some excellent open source projects: [PhaseNet](https://github.com/AI4EPS/PhaseNet), [EQTransformer](https://github.com/smousavi05/EQTransformer), [DiTing-FOCALFLOW](https://github.com/mingzhaochina/DiTing-FOCALFLOW), [MagNet](https://github.com/smousavi05/MagNet), [Deep-Bays-Loc](https://github.com/smousavi05/Deep-Bays-Loc), [PNW-ML](https://github.com/niyiyu/PNW-ML), and [SeisBench](https://github.com/seisbench/seisbench).


## License
Copyright S.Li et al. 2023. Licensed under an MIT license.



