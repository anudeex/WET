# WET
Code of our paper - "WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermarks"
**arXiv (Pre-print) version: [link]()**

<br />

<div align="center">
<img width="1150" alt="Screenshot 2024-08-26 at 17 07 08" src="https://github.com/user-attachments/assets/5ab91a7b-d999-41aa-89b4-e752ebf5e8d9">
  
**Overview of paraphrasing attack**
</div>

<br />


<div align="center">
<img width="1263" alt="Screenshot 2024-08-26 at 17 07 20" src="https://github.com/user-attachments/assets/bfc4cdb5-baa1-42e1-8a19-b22830e8e8c3">
  
**Overview of WET watermarking technique.**
</div>
<br />


## Abstract
Embeddings-as-a-Service (EaaS) is a service offered by large language model (LLM) developers to supply embeddings generated by LLMs. Previous research suggests that EaaS is prone to imitation attacks---attacks that clone the underlying EaaS model by training another model on the queried embeddings. As a result, EaaS watermarks are introduced to protect the intellectual property of EaaS providers. In this paper, we first show that existing EaaS watermarks can be removed by paraphrasing when attackers clone the model. Subsequently, we propose a novel watermarking technique that involves linearly transforming the embeddings, and show that it is empirically and theoretically robust against paraphrasing.


### 
Our code is based on the work of [WARDEN](https://github.com/anudeex/WARDEN)

## Citing

```
```
