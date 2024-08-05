---
layout: post
title:  "The Pseudoinverse Method to Fine-Tune LLMs for Binary Classification"
date:   2024-08-02 15:26:31 +0200
katex: true
---

Large Language Models (LLMs) are excellent baseline models for **zero-shot** classification, i.e. without any labeled examples. Most often than not, one has a small labeled dataset representative of the task, and is interested in improving the performance over this baseline. 

The quickest way to use this labeled data is to provide the labeled examples as part of the classification prompt (**few-shot** classification). The next step one can take is to fine-tune the model on the labeled dataset. There are several ways to fine-tune LLMs: e.g. optimizing all network parameters, optimizing only the parameters of the final layer, or freezing all parameters but introduce a new smaller set of tunable parameters [1]. Training all parameters requires significant GPU memory for large models, throwing aways the last layer and seems wasteful if the baseline model already performs better than random.

<div style="text-align: center;">
  <img 
    src="/diagrams/linear-adaptation/linear-adaptation.png" 
    alt="Learned Linear Transformation"
    style="width: 70%;"
  >
  <p><strong>Figure 1:</strong><em>The learned transformation W (green) is applied in parallel to the existing linear layer (blue), preserving the existing knowledge by the model. Both are the summed to make the logits, which are passed to the softmax function.</em></p>
</div>

In this post, I focus on binary classification. This means that the model must only answer ``yes/no`` or ``0/1`` to the prompt. The multiclass classification case can be reduced to a set binary classification problems. Furthermore, the binary case is easier to anaylze with metrics like the F1 score. After fine-tuning we can also interpret the changed model by looking at the answers that flipped between ``yes/no`` and vice-versa [2]. We will see that the binary problem structure can be leveraged for compute efficient fine-tuning. Concretely, one can fine-tune the LLM by adding only one additional learned linear transformation $$W$$ before the softmax, as shown in figure 1. This transformation accepts a closed-form solution given by the Moore-Penrose Inverse. The fine-tuned model is mathematically equivalent to adding a linear layer before the logits and training it with gradient descent, but must perform a lot less computation.

### Formal Derivation
In the original language model, the probability vector $$y$$ has the same size of the vocabulary, and is given by:

$$
\begin{aligned}
p(y | x_{1:t}) 
&= s(\text{logits}_t) \\
&= s(A z_t + b)
\end{aligned}
$$

where $$s()$$ is the softmax function, $$z_t$$ is the hidden state for the last token (dimension $$d$$), and $$A$$ and $$b$$ are the weights and bias of the last linear layer. The loss for this model is defined as the distance between these probabilites and our true labels, which is a set of binary labels $$Y = \{y_1, y_2, ..., y_N \}$$, $$Y \in \mathbb{R}^{(N, 2)}$$

During fine-tuning, we modify the probabilities that the LLM assigns to the tokens for ``yes/no`` by changing the logits that are passed to the softmax function. Here is where the learned transformation $$W$$ comes into play. We will tweak $$W$$ to approximate the dataset $$Y$$. The fine-tuned probabilities $$p'$$ are given by:

$$
\begin{aligned}
p'(y | x_{1:t}) &= s(logits_t + W z_t) \\
\implies \log p' &= logits_t + W z_t
\end{aligned}
$$

In vectorized form (one row per datapoint $$N$$) this can be written as:

$$
\begin{aligned}
\log P' &= L + Z_t W^T
\end{aligned}
$$

where:
* predicted probabilities: $$P' \in \mathbb{R}^{(N,V)}$$
* stacked logits: $$L \in \mathbb{R}^{(N,V)}$$, and 
* stacked hidden states: $$Z_t \in \mathbb{R}^{(N,d)}$$.

We can enforce compliance to the binary task by only evaluating the logits for the ``yes/no`` tokens. This reduces the size of the probability vector and of the logits to only 2. 



## References
[1] Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).

[2] Dutta, Abhinav, et al. "Accuracy is Not All You Need." arXiv preprint arXiv:2407.09141 (2024).
