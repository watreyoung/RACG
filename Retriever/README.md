## Retriever

### 1.1 Text Retrieval

There are two text retrieval algorithms in our experiments. They retrieve similar natural language descriptions with natural language input, and select the corresponding code snippets as the retrieved results.

#### BM25

[bm25.py](./Retriever/bm25.py) is based on rank_bm25 package to finsh the text retrieval task, so you should install the package and run the script.

#### RetroMAE

Official impletation of RetroMAE can be found in [https://github.com/staoxiao/RetroMAE](https://github.com/staoxiao/RetroMAE), and you can download the model from [https://huggingface.co/Shitao/RetroMAE_MSMARCO_finetune](https://huggingface.co/Shitao/RetroMAE_MSMARCO_finetune).

In our experimental setup, we directly employ RetroMAE for inference because of the absence of labeled natural language data to train it for text retrieval tasks. Therefore, you should transform the original dataset (i.e., \<Natural Language Description, Code Snippet\>)  into the format that RetroMAE can process.

The running details can be found in [RetroMAE](./RetroMAE).

### 1.2 Code Search

#### CodeBERT

Official impletation of CodeBERT can be found in [https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/codesearch](https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/codesearch), and you can download the model from [https://huggingface.co/microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base).

Due to the adaptation of [REDCODER](https://github.com/rizwan09/REDCODER) for retrieval tasks in the code generation domain, we utilized the codebert implementation from REDCODER instead of directly using the official implementation of CodeBERT.

The running details can be found in [CodeBERT](./CodeBERT).

#### UniXcoder

We use the official impletation of UniXcoder, which can be found in [https://github.com/microsoft/CodeBERT/tree/master/UniXcoder/downstream-tasks/code-search](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder/downstream-tasks/code-search), and you can download the model from [https://huggingface.co/microsoft/unixcoder-base](https://huggingface.co/microsoft/unixcoder-base).

The running details can be found in [UniXcoder](./UniXcoder).

#### CoCoSoDa

We use the official impletation of CoCoSoDa, which can be found in [https://github.com/DeepSoftwareAnalytics/CoCoSoDa](https://github.com/DeepSoftwareAnalytics/CoCoSoDa), and you can download the model from [https://huggingface.co/DeepSoftwareAnalytics/CoCoSoDa](https://huggingface.co/DeepSoftwareAnalytics/CoCoSoDa).

The running details can be found in [CoCoSoDa](./CoCoSoDa).