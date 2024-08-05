---
layout: post
title:  "The Pseudoinverse Method to Fine-Tune LLMs for Binary Classification"
date:   2024-08-02 15:26:31 +0200
---

<div style="text-align: center;">
  <img src="/diagrams/linear-adaptation/linear-adaptation.png" alt="Learned Linear Transformation">
  <p><em>Figure 1: The learned transformation W (green) is applied in parallel to the existing linear layer (blue), preserving the existing knowledge by the model. Both are the summed to make the logits, which are passed to the softmax function.</em></p>
</div>

Large Language Models (LLMs) are excellent baseline models for **zero-shot** classification, i.e. without any labeled examples. Most often than not, one has a small labeled dataset representative of the task, and is interested in improving the performance over this baseline. 

The quickest way to use this labeled data is to provide the labeled examples as part of the classification prompt (**few-shot** classification). The next step one can take is to fine-tune the model on the labeled dataset. There are several ways to fine-tune LLMs: e.g. optimizing all network parameters, optimizing only the parameters of the final layer, or freezing all parameters but introduce a new smaller set of tunable parameters [1]. Training all parameters requires significant GPU memory for large models, throwing aways the last layer and seems wasteful if the baseline model already performs better than random.

In this post, I focus on binary classification. This means that the model must only answer ``yes/no`` or ``0/1`` to the prompt. The multiclass classification case can be reduced to a set binary classification problems. Furthermore, the binary case is easier to anaylze with metrics like the F1 score. After fine-tuning we can also interpret the changed model by looking at the answers that flipped between ``yes/no`` and vice-versa [2]. We will see that the binary problem structure can be leveraged for compute efficient fine-tuning. Concretely, one can fine-tune the LLM by adding only one additional learned linear transformation $$W$$ before the softmax. This transformation accepts a closed-form solution given by the Moore-Penrose Inverse. The fine-tuned model is mathematically equivalent to adding a linear layer before the logits and training it with gradient descent, but must perform a lot less computation.


## References
[1] Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).

[2] Dutta, Abhinav, et al. "Accuracy is Not All You Need." arXiv preprint arXiv:2407.09141 (2024).
