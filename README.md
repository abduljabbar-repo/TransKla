**TransKla: A Local-Global Cross-Attention Based Transformer Approach for Prediction of Lysine Lactylation**

Lysine lactylation (Kla) is a novel post-translational modification that bridges metabolic flux with epigenetic signaling. Dysregulation of lactylation disrupts multiple biological pathways, driving pathological states including oncogenesis, neural hyperexcitability, and immune dysfunction. Although wet-lab experiments are considered the gold standard, they are expensive and laborious. While computational methods have contributed to alternative solutions, they often fail to capture the unique biochemical properties of lactylation or integrate local sequence patterns with global protein context. To address these challenges, we present TransKla, a novel transformer-based framework that integrates key physicochemical features (charge and hydrophobicity) and sequence embeddings into a unified representation. The model combines local 41-residue context with global protein representations via cross-attention to capture long-range dependencies. Extensive ablation studies and a comprehensive regularization strategy validate our architectural choices and prevent overfitting. TransKla results in an AUPRC of 0.891, an AUC of 0.891, and an accuracy of 0.805 on the validation test, an increase in performance of 3.4%, 2.7%, and 3.1% on test data, respectively, and significantly outperforms state-of-the-art methods on the independent human Kla dataset. Moreover, our model uses ∼4.03 million parameters, much fewer than models that leverage large language models (LLMs). TransKla stands out as a useful tool, enabling accurate lactylation predictions with minimal computational resources.

# Requirements

	To create an environment, the following libraries are required:
		
	torch
	numpy
	pandas
	scikit-learn

## Dataset and Code Availability

A standalone package is provided to support independent testing and reproducibility.

### Dataset 1: Benchmark Dataset
The `Benchmark Dataset` folder contains the filtered lysine lactylation sites used in this study, including the training, validation, and independent test sets.

### Dataset 2: Human Kla Dataset
For comparison with state-of-the-art methods, TransKla was evaluated on the Human Kla dataset used by PBertKla. The dataset is available at:

https://github.com/laihongyan/PBertKla/tree/main/Data

### Model Implementation
The implementation of the TransKla model is provided in:

```bash
train_TransKla.py



