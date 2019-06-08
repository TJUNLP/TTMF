# Triple Trustworthiness Measurement for Knowledge Graph

This is the source code for the paper ''Triple Trustworthiness Measurement for Knowledge Graph'' accepted by WWW2019.

The Knowledge graph (KG) uses the triples to describe the facts in the real world. It has been widely used in intelligent analysis and applications. However, possible noises and conflicts are inevitably introduced in the process of constructing. And the KG based tasks or applications assume that the knowledge in the KG is completely correct and inevitably bring about potential deviations. In this paper, we establish a knowledge graph triple trustworthiness measurement model that quantify their semantic correctness and the true degree of the facts expressed. The model is a crisscrossing neural network structure. It synthesizes the internal semantic information in the triples and the global inference information of the KG to achieve the trustworthiness measurement and fusion in the three levels of entity level, relationship level, and KG global level. We analyzed the validity of the model output confidence values, and conducted experiments in the real-world dataset FB15K (from Freebase) for the knowledge graph error detection task. The experimental results showed that compared with other models, our model achieved significant and consistent improvements.

## Requirements
* Keras
* python 3.x

## Usage
We perform experiments on the FB15K, which is a typical benchmark knowledge graph extracted from Freebase. 

* Model1.py  --Program Main Entry

* PrecessData.py  --Pre-process data for Model1.py.

* search.py  --To construct the directed graphs centered on each head entity h.

* pagerank.py  --Iterating the flow of resources in each graph.

* ResourceRankConfidence.py  --To get entity-level Estimator values RR (h, t), corresponding to Section 3.1 of paper.

* TransConfidence.py  --To get relation-level Estimator values P(E(h,r,t)), corresponding to Section 3.2 of paper.

* SearchPaths2.py  --To select reachable paths, corresponding to Section 3.3 of paper.

### You can get more intermediate data that has been generated, from https://pan.baidu.com/s/1KZpOe3xUQTUnOxC1QJRPLQ  pwd:8xqd


## Citation
If you use the code, please cite this paper:

Shengbin Jia, Yang Xiang, and Xiaojun Chen. 2019. Triple Trustworthiness Measurement for Knowledge Graph. In Proceedings of the 2019 World Wide Web Conference (WWW ’19), May 13–17, 2019, San Francisco, CA, USA. ACM, New York, NY, USA, 7 pages. 

Contact: shengbinjia@tongji.edu.cn
