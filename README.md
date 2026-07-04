**TransKla: A Local-Global Cross-Attention Based Transformer Approach for Prediction of Lysine Lactylation**

Lysine lactylation (Kla) is a novel post-translational modification that bridges metabolic flux with epigenetic signaling. Dysregulation of lactylation disrupts multiple biological pathways, driving pathological states including oncogenesis, neural hyperexcitability, and immune dysfunction. Although wet-lab experiments are considered the gold standard, they are expensive and laborious. While computational methods have contributed to alternative solutions, they often fail to capture the unique biochemical properties of lactylation or integrate local sequence patterns with global protein context. To address these challenges, we present TransKla, a novel transformer-based framework that integrates key physicochemical features (charge and hydrophobicity) and sequence embeddings into a unified representation. The model combines local 41-residue context with global protein representations via cross-attention to capture long-range dependencies. Extensive ablation studies and a comprehensive regularization strategy validate our architectural choices and prevent overfitting. TransKla results in an AUPRC of 0.891, an AUC of 0.891, and an accuracy of 0.805 on the validation test, an increase in performance of 3.4%, 2.7%, and 3.1% on test data, respectively, and significantly outperforms state-of-the-art methods on the independent human Kla dataset. Moreover, our model uses ∼4.03 million parameters, much fewer than models that leverage large language models (LLMs). TransKla stands out as a useful tool, enabling accurate lactylation predictions with minimal computational resources. The dataset and source code used in this study are freely accessible at https://github.com/abduljabbar-repo/TransKla.git.

# Requirements

	To create an environment, the following libraries are required:
		
	torch
	numpy
	pandas
	scikit-learn

# Standalone package is provided for independent testing 

Dataset 1: The dataset used for training, validation, and independent testing
	   Benchmark_dataset.fasta

Dataset 2: The state-of-the-art studies are compared against the Human Kla dataset, whose link is given below
           https://github.com/laihongyan/PBertKla/tree/main/Data

You can find the implementation of the model in train_TransKla.py file



