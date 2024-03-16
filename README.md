# ECE C147 Final Project

Muhan Zhang | Yubo Zhang | Yuxuan Qi | Xuyang Zhou

## Directory Structure

The training and testing routines of all models are in their respective jupyter notebooks.

- CNN.ipynb: Vanilla CNN
- CNN_RNN.ipynb: Hybrid CNN+RNN
- CNN_Transformer.ipynb: Visual Transformer
    - The actual implementation of the Visual Transformer is in the `models` directory
- resnet.ipynb: ResNet
- transformer.ipynb: Pure Transformer. This is not included in our final models, because we developed the visual
  transformer model based on this.

The trained models are saved in the `models` directory.

Under the root directory, `prepare_data.py` and `prepare_frequency_data.py` are used to prepare the data for the models.

## How to Run

First, install the required packages by running `pip install -r requirements.txt`. Then, download the EEG data and put it in the `data` directory. Finally, run the jupyter notebooks to train and test the models.
