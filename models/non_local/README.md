# Non-Local Neural Networks

Since our input data has long sequences (L=2000=20[sec]\*100[Hz]), capturing long-range dependencies is important. To this end, we applied **Non-Local Neural Networks** [[1]](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html) for this project.
The Non-Local(NL) operation was inspired by NLM noise filter, described in Wikipedia (https://en.wikipedia.org/wiki/Non-local_means):

> Unlike "local mean" filters, which take the mean value of a group of pixels surrounding a target pixel to smooth the image, non-local means filtering takes a mean of all pixels in the image, weighted by how similar these pixels are to the target pixel.

The Non-Local(NL) operation, in action recognition, computes weight matrix representing relationships between all different positions in both space and time. The authors of [[1]](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html) insist that NL operation has advantages of:

- compared to recurrent local convolutions, NL can capture **long-range dependencies** directly by computing interactions between any two positions, regradless of their positional distance
- computationally efficient
- generic; can cope with variable input size
  - 1D, 2D, 3D conv version of NL can easilty be implemented

We deflated 2D ResNet into 1D and placed some NL blocks after residual blocks, and proceed 10 epochs of training.
Currently, this codebase supports the following models:

- R18
- ~~R34, R50, R101~~

R34, R50, R101 are not included for memory efficiency.

### R18

| # of NLBlocks[`res1`,`res2`,`res3`,`res4`,`res5`] |   AUPRC    |   AUROC    |                                           Pretrained model                                           |
| :-----------------------------------------------: | :--------: | :--------: | :--------------------------------------------------------------------------------------------------: |
|                     0,0,0,0,1                     |  0.46943   |  0.92330   | [link](https://www.dropbox.com/s/woh99y2hll1mlqv/irCSN_152_Sports1M_from_scratch_f99918785.pkl?dl=0) |
|                     0,0,2,3,0                     | **0.4877** | **0.9239** | [link](https://www.dropbox.com/s/woh99y2hll1mlqv/irCSN_152_Sports1M_from_scratch_f99918785.pkl?dl=0) |

## TODO

- [x] ~~NL position test code~~
- [ ] More regularization methods on abstracted features

* Dropout
* Maxout[[2]](https://arxiv.org/abs/1302.4389)
* Stochastic pooling[[3]](https://arxiv.org/abs/1301.3557)
* BERT pooling[[4]](https://arxiv.org/abs/2008.01232)

## References

1. Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He. **Non-Local Neural Networks.** CVPR 2018.
2. Goodfellow, Ian, et al. "Maxout networks." International conference on machine learning. PMLR, 2013.
3. Zeiler, et al. "Stochastic pooling for regularization of deep convolutional neural networks." arXiv preprint (2013).
4. Kalfaoglu, et al. "Late Temporal Modeling in 3D CNN Architectures with BERT for Action Recognition." arXiv preprint (2020).
