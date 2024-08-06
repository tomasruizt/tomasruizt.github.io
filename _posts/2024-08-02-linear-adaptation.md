---
layout: post
title:  "A Closed-Form Solution to Linearly Fine-Tune LLMs for Binary Classification"
date:   2024-08-02 15:26:31 +0200
katex: true
---

# Summary

In this post I show how to linearly fine-tune a large language model (LLM) using a closed-form solution, based on the Moore-Penrose Inverse. I will focus on the special case of binary classification because it makes this approach computationally efficient. The new linear transformation (green) $$W$$ is shown in figure 1.

<div style="text-align: center;">
  <img 
    src="/diagrams/linear-adaptation/linear-adaptation.png" 
    alt="Learned Linear Transformation"
    style="width: 70%;"
  >
  <p><strong>Figure 1:</strong><em>The learned transformation W (green) is applied in parallel to the existing linear layer (blue), preserving the existing knowledge of the model. Both are the summed to make the logits, which are passed to the softmax function.</em></p>
</div>

# About Fine-Tuning

Large Language Models (LLMs) are great baseline models for **zero-shot** classification, i.e. without any labeled examples. However, one often has a small labeled dataset and is interested in improving the performance over this baseline. In the **few-shot** setting, some labeled examples are provided in the prompt for the model to learn in context. To improve upon this setting, the next step is to **fine-tune** the model on the labeled dataset.

There are different ways to fine-tune LLMs. For example: optimizing all network parameters, optimizing only the parameters of the final layer, or freezing all parameters but introduce a new smaller set of tunable parameters [1]. In this blog post I focus on the simple case of fine-tuning the last linear transformation, because I'm interested in interpreting the changes in individual logits and probabilities later on.

# Binary Classification

I also focus on binary classification specifically. This means that the model must only answer ``yes/no`` or ``0/1`` to the prompt. The reason is that this setting is easier to interpret with metrics like precision, recall or the $$F1$$ score. Furthermore, in the binary case we can interpret how the fine-tuning procedure affected the model by inspecting the answers that flipped between ``yes/no`` and vice-versa [2]. The multiclass classification case can be reduced to a set binary classification problems.

In terms of computation, we will see that the problem structure of binary classification can be leveraged to compute a closed-form solution efficiently. Concretely, one can fine-tune the LLM by adding only one additional learned linear transformation $$W$$ before the softmax, as shown in figure 1. This transformation accepts a closed-form solution given by the Moore-Penrose Inverse. The fine-tuned model is mathematically equivalent to adding a linear layer before the logits and training it with gradient descent, but must perform a lot less computation.

### Formal Derivation
In the original language model, the probability vector $$y$$ has the same size of the vocabulary, and is given by:

$$
\begin{aligned}
\underbrace{p(y | x_{1:t})}_{(1,V)}
&= \text{softmax}(\underbrace{\text{logits}_t}_{(1,V)}) \\
&= \text{softmax}(\underbrace{z_t}_{(1,d)} \underbrace{A}_{(d,V)} + \underbrace{b}_{(1,V)})
\end{aligned}
$$

where $$z_t$$ is the hidden state for the last token, and $$A$$ and $$b$$ are the weights and bias of the last linear layer. The loss for this model is defined as the distance between these probabilites and our true labels, which is a set of binary labels $$Y = \begin{bmatrix} y_1 \\ ... \\ y_N \end{bmatrix} \in \mathbb{R}^{(N, 2)}$$

During fine-tuning, we modify the probabilities that the LLM assigns to the tokens for ``yes/no`` by changing the logits that are passed to the softmax function. Here is where the learned transformation $$W$$ comes into play. We will tweak $$W$$ to approximate the dataset $$Y$$ with the fine-tuned probabilities $$p'$$, which are given by:

$$
\begin{aligned}
p'(y | x_{1:t}) &= \text{softmax}(\text{logits}_t + z_t W) \\
\implies \underbrace{\log p'}_{(1, V)} 
&= \underbrace{\text{logits}_t}_{(1,V)} + \underbrace{z_t}_{(1,d)} \underbrace{W}_{(d,V)}
\end{aligned}
$$

In vectorized form (one row per datapoint $$N$$) this can be written as:

$$
\begin{aligned}
\underbrace{\log P'}_{(N,V)} 
&= \underbrace{L_t}_{(N,V)} + \underbrace{Z_t}_{(N,d)} \underbrace{W}_{(d,V)}
\end{aligned}
$$

Solving for $$W$$ exactly only works for non-singular matrices $$Z_t$$. For general matrices, this problem is solved by minimizing the is a squared distance:

$$W = \argmin_W || Z_t W - (\log P' - L_t) ||^2_2 $$

This is a least squares problem, whose solution is given by the **Moore-Penrose Inverse**:

$$W = (Z_t^T Z_t)^{-1} Z_t^T (\log P' - L_t)$$

Or equivalently, by solving the following linear system with $$V$$ columns.

$$W = \text{linsolve}(\underbrace{Z_t^T Z_t}_{(d,d)}, \space \underbrace{Z_t^T (\log P' - L_t)}_{(d,V)})$$

Solving $$V$$ linear systems of order $$O(d)$$ is prohibitively expensive ($$V=128k, d=4k$$ in LLama3 8B). However, we can exploit the structure of the binary classification problem, by only evaluating the logits $$L_t$$ and probabilities $$P'$$ for the ``yes/no`` tokens. This reduces the size of the probability matrix $$P'$$ from $$(N,V)$$ to $$(N,2)$$, and the size of the learned matrix $$W$$ from $$(d,V)$$ to $$(d,2)$$. 

As a result, we need to solve only 2 linear systems, each with a dimension that scales as $$O(d)$$, and is constant in the vocabulary size $$V$$ and in the number of datapoints in our traning dataset $$N$$. Additionally, the output of the fine-tuned model the compliant by design, as it cannot output any other logits than for ``yes/no``.

### Inference
At inference time, the matrix $$W$$ stays constant, while the logits change for each input.

$$p'(y|x_{1:t}) = \text{softmax}(z_t(x_{1:t})(A + W) + b)$$

## References
[1] Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).

[2] Dutta, Abhinav, et al. "Accuracy is Not All You Need." arXiv preprint arXiv:2407.09141 (2024).
