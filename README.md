# SleepXViT
- Explainable vision transformer for automatic visual sleep staging on multimodal PSG signals


## Framework Overview
![Framework_Overview](https://github.com/user-attachments/assets/39b3f782-9358-4b6c-89c3-65187765a5c5)


## Dataset
- ### [KISS Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=210)
    Jaemin Jeong, Wonhyuck Yoon, Jeong-Gun Lee, Dongyoung Kim, Yunhee Woo, Dong-Kyu Kim, Hyun-Woo Shin, __Standardized image-based polysomnography database and deep learning algorithm for sleep-stage classification__, *Sleep*, 2023;, zsad242, https://doi.org/10.1093/sleep/zsad242 (Only Avaliable at Restricted Area)
    ![KISS Dataset](https://github.com/user-attachments/assets/9a64aca5-02fd-49a0-872a-4c8d7d58db15)

- ### [SHHS Dataset](https://www.sleepdata.org/datasets/shhs)
    Zhang GQ, Cui L, Mueller R, Tao S, Kim M, Rueschman M, Mariani S, Mobley D, Redline S. The National Sleep Research Resource: towards a sleep data commons. J Am Med Inform Assoc. 2018 Oct 1;25(10):1351-1358. doi: 10.1093/jamia/ocy064. PMID: 29860441; PMCID: PMC6188513.
  Quan SF, Howard BV, Iber C, Kiley JP, Nieto FJ, O'Connor GT, Rapoport DM, Redline S, Robbins J, Samet JM, Wahl PW. The Sleep Heart Health Study: design, rationale, and methods. Sleep. 1997 Dec;20(12):1077-85. PMID: 9493915.


## Model Architecture
### Intra-Epoch ViT
![7_a_intra_epoch](https://github.com/user-attachments/assets/52b4ef9e-ab66-4d82-a713-a014e23f5134)

### Inter-Epoch ViT
![7_b_inter_epoch](https://github.com/user-attachments/assets/7fefc057-c0ca-4b3b-82de-f38a09af0d14)


## Interpretability
### Inter-Epoch Interpretability
- Adopted Transformer Interpretability Beyond Attention Visualization (CVPR 2021) [[Github]](https://github.com/hila-chefer/Transformer-Explainability)
![Picture2](https://github.com/user-attachments/assets/1f9a1a79-540a-4067-b994-c1082c0ad239)

### Intra-Epoch Interpretability
- Adopted same method as above.
- Using sliding window scheme and aggregated 15 softmax values to predict one Epoch.
![InterEpoch](https://github.com/user-attachments/assets/b30fb910-2a6e-43dc-8f43-67644ded9406)

