## SoftAlign: End-to-End Structural Alignment for Protein Data

SoftAlign is an advanced alignment method designed to efficiently compare 3D protein structures. By leveraging structural information directly, SoftAlign provides an end-to-end alignment process, allowing for both highly accurate alignments and efficient computations. The method uses the 3D coordinates of protein pairs, transforming them into feature vectors through a retrained encoder of ProteinMPNN. This similarity matrix is then aligned using two strategies: a differentiable Smith-Waterman method and a novel softmax-based pseudo-alignment approach.

Our results demonstrate that SoftAlign is able to recapitulate TM-align results while being faster and more accurate than alternative tools like Foldseek. While not the fastest alignment method available, SoftAlign excels in precision and is well-suited for integration with other pre-filtering methods. Notably, the softmax-based alignment shows superior sensitivity for structure similarity detection compared to traditional methods.

SoftAlign also introduces a novel pseudo-alignment method based on softmax. This approach can be integrated into other models and architectures, even those not inherently focused on structural information. For a more detailed description of the method, please refer to the full paper [here](https://github.com/jtrinquier/SoftAlign).

---

## 🔬 Google Colab Notebooks

To facilitate ease of use and reproducibility, we provide three Google Colab notebooks:

1. **Training Notebook**: Reproduces the training process with the same train-test split as described in our paper.  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/jtrinquier/SoftAlign/blob/main/Colab/SoftAlign_training.ipynb)  
   [SoftAlign_training.ipynb](https://github.com/jtrinquier/SoftAlign/blob/main/Colab/SoftAlign_training.ipynb)

2. **Inference Notebook**: Allows users to experiment with the learned aligners on their own data.  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/jtrinquier/SoftAlign/blob/main/Colab/COLAB_SoftAlign.ipynb)  
   [COLAB_SoftAlign.ipynb](https://github.com/jtrinquier/SoftAlign/blob/main/Colab/COLAB_SoftAlign.ipynb)

3. **All-vs-All Search Notebook**: Performs an all-vs-all search within the SCOPE 40 dataset. *Note: This notebook is still in development.*  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/jtrinquier/SoftAlign/blob/main/Colab/SCOPE_SEARCH_SoftAlign.ipynb)  
   [SCOPE_SEARCH_SoftAlign.ipynb](https://github.com/jtrinquier/SoftAlign/blob/main/Colab/SCOPE_SEARCH_SoftAlign.ipynb)



