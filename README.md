# Cross subject EEG emotion recognition through NN
This code corresponds to the work "Cross-Subject EEG-Based Emotion Recognition Through Neural Networks With Stratified Normalization" submitted to Frontiers in Neuroscience.

The structure is as follows:
* `create_features.ipynb`: Code to compute the Welch, multitaper and DE features
* `cross_subject_welch.ipynb`: Code to compute the accuracies for the Welch features
* `cross_subject_multitaper.ipynb`: Code to compute the accuracies for the multitaper features
* `cross_subject_de.ipynb`: Code to compute the accuracies for the DE features

To download the [SEED dataset](https://bcmi.sjtu.edu.cn/home/seed/seed.html), please follow the next instructions [instructions](https://bcmi.sjtu.edu.cn/home/seed/downloads.html#seed-access-anchor).

# How To

## Installing dependencies
Use Python 3.9 and install dependencies with venv:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Creating features

After downloading the data, place all the preprocessing files `mat` files under `data/dataset/SEED/Preprocessed_EEG`.

# Contact email:
Javier Fdez: javier.f3rnand3z@gmail.com
