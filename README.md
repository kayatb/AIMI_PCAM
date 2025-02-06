# Comparing Explainable Models and Black-Box Models for Automatic Metastatic Tissue Detection

Our report can be found [here](https://github.com/kayatb/AIMI_PCAM/blob/main/Report_PatchCamelyon.pdf).

Project repository for the course Artificial Intelligence for Medical Imaging and the [PatchCamelyon Challenge](https://patchcamelyon.grand-challenge.org/).


![metastatic](https://github.com/kayatb/AIMI_PCAM/blob/main/imgs/metastatic.png)

![non-metastatic](https://github.com/kayatb/AIMI_PCAM/blob/main/imgs/non-metastatic.png)

## Abstract
Manual metastatic tissue detection is a time-consuming task and it would be beneficial to (partially) automate this process. This is a high-stakes environment, which means that not only should this task be done with high performance, but the decision-making process should also be interpretable. In this work we evaluate various black-box models: ResNet, ResNeSt and our own group equivariant CNN model. Furthermore, we train an inherently explainable model, which uses slot attention to generate its own explanations. We found no significant improvement from using equivariance compared to the non-equivariant baseline. The interpretable models provide useful visualisations while achieving comparable performance to the non-explainable black-box models. We argue for the importance of explainable AI in such a high-stakes environment and show that having interpretable models is a viable direction for automatic metastatic tissue detection.
