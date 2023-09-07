# SAMB-TransformerUDA
Semantic-adaptive Message Broadcasting for Transformer-based UDA
  

[Xin Li](http://home.ustc.edu.cn/~lixin666/), [Cuiling Lan](https://scholar.google.com/citations?user=XZugqiwAAAAJ&hl=en), [Guoqiang Wei](https://scholar.google.com/citations?user=TxeZUTgAAAAJ&hl=en), [Zhibo Chen](https://scholar.google.com/citations?user=1ayDJfsAAAAJ&hl=en)

University of Science and Technology of China (USTC), Microsoft Research Asia (MSRA),

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2212.02739)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-aware-message-broadcasting-for/unsupervised-domain-adaptation-on-office-home)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-office-home?p=semantic-aware-message-broadcasting-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-aware-message-broadcasting-for/unsupervised-domain-adaptation-on-visda2017)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-visda2017?p=semantic-aware-message-broadcasting-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-aware-message-broadcasting-for/unsupervised-domain-adaptation-on-domainnet-1)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-domainnet-1?p=semantic-aware-message-broadcasting-for)

## 📌 News!!!
- [x] | 07/09/2023 | The code for the first version of training strategies was released. 
- [ ] The second version of the training strategy will be updated with better performance.


## 🌟 Motivation
<p align="center">
  <img src="https://github.com/lixinustc/SAMB-TransformerUDA/blob/main/figs/SAMB.png" width="85%" height="85%">
</p>

## Usage

Please replace the base_trainer.py in the framework of [ToAlign](https://github.com/microsoft/UDA) with this file and the model part with your model. 


## Cite US
Please cite us if this work is helpful to you.


```
@article{li2022semantic,
  title={Semantic-aware Message Broadcasting for Efficient Unsupervised Domain Adaptation},
  author={Li, Xin and Lan, Cuiling and Wei, Guoqiang and Chen, Zhibo},
  journal={arXiv preprint arXiv:2212.02739},
  year={2022}
}
```


# Abstract
Vision transformer has demonstrated great potential
in abundant vision tasks. However, it also inevitably suffers
from poor generalization capability when the distribution shift
occurs in testing (i.e., out-of-distribution data). To mitigate this
issue, we propose a novel method, Semantic-aware Message
Broadcasting (SAMB), which enables more informative and
flexible feature alignment for unsupervised domain adaptation
(UDA). Particularly, we study the attention module in the vision
transformer and notice that the alignment space using one global
class token lacks enough flexibility, where it interacts information
with all image tokens in the same manner but ignores the rich
semantics of different regions. In this paper, we aim to improve
the richness of the alignment features by enabling semanticaware adaptive message broadcasting. Particularly, we introduce
a group of learned group tokens as nodes to aggregate the
global information from all image tokens, but encourage different
group tokens to adaptively focus on the message broadcasting to
different semantic regions. In this way, our message broadcasting
encourages the group tokens to learn more informative and
diverse information for effective domain alignment. Moreover,
we systematically study the effects of adversarial-based feature
alignment (ADA) and pseudo-label based self-training (PST) on
UDA. We find that one simple two-stage training strategy with
the cooperation of ADA and PST can further improve the adaptation capability of the vision transformer. Extensive experiments
on DomainNet, OfficeHome, and VisDA-2017 demonstrate the
effectiveness of our methods for UDA



## Acknowledgments
- [ToAlign](https://github.com/microsoft/UDA)
- [CDTrans](https://github.com/CDTrans/CDTrans)


