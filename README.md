# SAMB-TransformerUDA
Semantic-adaptive Message Broadcasting for Transformer-based UDA
  
> [**SAMB**](https://arxiv.org/abs/2212.02739), Xin Li, Cuiling Lan, Guoqiang Wei, Zhibo Chen,

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2212.02739)
![image](https://github.com/lixinustc/SAMB-TransformerUDA/blob/main/figs/SAMB.png)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-aware-message-broadcasting-for/unsupervised-domain-adaptation-on-office-home)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-office-home?p=semantic-aware-message-broadcasting-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-aware-message-broadcasting-for/unsupervised-domain-adaptation-on-visda2017)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-visda2017?p=semantic-aware-message-broadcasting-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-aware-message-broadcasting-for/unsupervised-domain-adaptation-on-domainnet-1)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-domainnet-1?p=semantic-aware-message-broadcasting-for)

## 📌 News!!!
- [x] | 07/09/2023 | The code for the first version of training strategies was released.
- [ ] The second version of the training strategy will be updated with better performance. 



## Cite US
Please cite us if this work is helpful to you.


```
@article{li2023learning,
  title={Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective},
  author={Li, Xin and Li, Bingchen and Jin, Xin and Lan, Cuiling and Chen, Zhibo},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
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



