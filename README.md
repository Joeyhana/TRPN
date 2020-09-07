# TRPN
## Introduction
A pytorch implementation of the IJCAI2020 paper "[Transductive Relation-Propagation Network for Few-shot Learning](https://www.ijcai.org/Proceedings/2020/0112.pdf)". The code is based on [Edge-labeling Graph Neural Network for Few-shot Learning](https://github.com/khy0809/fewshot-egnn)

**Author:** Yuqing Ma, Shihao Bai, Shan An, Wei Liu, Aishan Liu, Xiantong Zhen and Xianglong Liu

**Abstract:** Few-shot learning, aiming to learn novel concepts from few labeled examples, is an interesting and very challenging problem with many practical advantages. To accomplish this task, one should concentrate on revealing the accurate relations of the support-query pairs. We propose a transductive relation-propagation graph neural network (TRPN) to explicitly model and propagate such relations across support-query pairs. Our TRPN treats the relation of each support-query pair as a graph node, named relational node, and resorts to the known relations between support samples, including both intra-class commonality and inter-class uniqueness, to guide the relation propagation in the graph, generating the discriminative relation embeddings for support-query pairs. A pseudo relational node is further introduced to propagate the query characteristics, and a fast, yet effective transductive learning strategy is devised to fully exploit the relation information among different queries. To the best of our knowledge, this is the first work that explicitly takes the relations of support-query pairs into consideration in few-shot learning, which might offer a new way to solve the few-shot learning problem. Extensive experiments conducted on several benchmark datasets demonstrate that our method can significantly outperform a variety of state-of-the-art few-shot learning methods.

## Requirements
* Python 3
* Python packages
  - pytorch 1.0.0
  - torchvision 0.2.2
  - matplotlib
  - numpy
  - pillow
  - tensorboardX

An NVIDIA GPU and CUDA 9.0 or higher. 
