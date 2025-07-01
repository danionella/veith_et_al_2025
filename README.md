# Figure 3 & 4

This stand-alone notebook reproduces the plots shown in Figures 3 and 4 of the paper  
**"Algorithms underlying directional hearing in *Danionella c.* startle behavior."**

👉 For a static preview, [open the notebook's outputs on GitHub](https://github.com/danionella/veith_et_al_2025/blob/main/generate_figures_3_4.ipynb).

---

## Running the Notebook on Google Colab

You can execute the notebook using a free runtime hosted by Google Research. Simply click the badge below:

<a target="_blank" href="https://colab.research.google.com/github/danionella/veith_et_al_2025/blob/main/generate_figures_3_4.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

## Running the Notebook Locally

### Requirements

No specific hardware is required. The code runs on Windows, macOS, and Linux systems.

### Installation Instructions

1. Install [conda for Python 3.x](https://github.com/conda-forge/miniforge).
2. Navigate to the directory containing this file.
3. Run the following commands in your terminal:

    ```bash
    conda create -n veith2025 'python>=3.8' numpy scipy pandas matplotlib seaborn jupyter -c conda-forge
    conda activate veith2025
    jupyter lab generate_figures_3_4.ipynb
    ```

These steps typically take less than 10 minutes.

Alternatively, you can open the notebook in an existing Jupyter environment after installing the required packages:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`

---

## Data Availability

The behavioral dataset used for generating the plots is available at:

🔗 [https://gin.g-node.org/danionella/Veith_et_al_2025/src/master/behavior](https://gin.g-node.org/danionella/Veith_et_al_2025/src/master/behavior)
