# Seismogram Transformer (SeisT)

## Introduction
SeisT is a backbone network for seismic signal processing, which can be used for multiple seismic monitoring tasks such as earthquake detection, seismic phase picking, first-motion polarity classification, magnitude estimation, and back-azimuth estimation.

This repository also provides some baseline model implemented by Pytorch under `./models`, such as PhaseNet, EQTransformer, DitingMotion, MagNet, BAZ-Network, and distPT-Network. 

NOTE: The model weights included in this repository serve as the basis for performance evaluation in this paper.  They have been evaluated using identical training and testing data and a consistent training regimen, thereby affirming the architecture's validity.  Nevertheless, if you intend to employ these models in practical engineering applications, it is crucial to retrain the SeisT models with larger datasets to align with the specific demands of engineering applications.

## Usage
Training and testing tutorials will be completed soon.

## Citation

Paper: https://arxiv.org/abs/2310.01037

If you find this repo useful in your research, please consider citing:

```
@misc{li2023seist,
      title={Seismogram Transformer: A generic deep learning backbone network for multiple earthquake monitoring tasks}, 
      author={Sen Li and Xu Yang and Anye Cao and Changbin Wang and Yaoqi Liu and Yapeng Liu and Qiang Niu},
      year={2023},
      eprint={2310.01037},
      archivePrefix={arXiv},
      primaryClass={physics.geo-ph}
}
```

The baseline model used in this paper:

- **PhaseNet**<br/>
  *Zhu, W., & Beroza, G. C. (2019). PhaseNet: A deep-neural-network-based seismic arrival-time picking method. Geophysical Journal International, 216(1), 261-273.*

- **EQTransformer**<br/>
  *Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., & Beroza, G. C. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature communications, 11(1), 3952.*

- **DiTingMotion**<br/>
  *Zhao, M., Xiao, Z., Zhang, M., Yang, Y., Tang, L., & Chen, S. (2023). DiTingMotion: A deep-learning first-motion-polarity classifier and its application to focal mechanism inversion. Frontiers in Earth Science, 11, 1103914.*

- **MagNet**<br/>
  *Mousavi, S. M., & Beroza, G. C. (2020). A machine‐learning approach for earthquake magnitude estimation. Geophysical Research Letters, 47(1), e2019GL085976.*

- **BAZ-Network** & **distPT-Network**<br/>
  *Mousavi, S. M., & Beroza, G. C. (2020). Bayesian-Deep-Learning Estimation of Earthquake Location From Single-Station Observations. IEEE Transactions on Geoscience and Remote Sensing, 58(11), 8211-8224.*


## License
Copyright S.Li et al. 2023. Licensed under an MIT license.



