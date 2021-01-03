# CoViD19Cat  :mask:

This repository contains the code used to perform the experiments in the `Covid19Cat: Covid Impact Prediction using Machine Learning Models` project.

> The aim of this document is to analyze the performance of various regression models on the prediction of the evolution of COVID-19 pandemic in Catalonia. A new data set will be created from the available open-source information and it will be tested on a variety of configurations, involving the use of data augmentation. The studied models in this project are Multi-Layer Perceptron (MLP), Support Vector Regressor (SVR), Adaboost, Random Forest Regressor (RFR) and a Recurrent Neural Network architectures based on LSTM units to explore temporality.

***

## Requirements

- Python >3.6
- scikit-learn
- sodapy==2.1.0
- pandas
- matplotlib
- numpy
- unidecode

## How to run :man_running: :snake:

Create a python environment with the required packages:

```bash
conda create -n covid3.7 python=3.7
conda activate covid3.7
pip install -r requirements.txt
```

To execute the SVR, RF, MLP and Adaboost GridSearch experiments:

```bash
python src/main.py
```

The script will generate a log file in the folder `./log` with the details. Also in the `./images` folder, an image with the full inference of the data will be saved.

To execute the lstm experiments:

```bash
cd src/
python lstm.py <lstm/lstm-nohidden/blstm/blstm-nohidden>
```

This will generate the best models in `h5` format and store them into the `src/checkpoints` directory. Additionally, some plots will also be saved into the `images/` folder.

To execute the 3 feature model with future day prediction:

```bash
cd src/
python lstm_future_prediction.py blstm3feats <number_of_days>
```

This will generate the plots into the `images/` folder.

## Databases and tokens associated

### Registre de casos de COVID-19 realitzats a Catalunya. Segregació per sexe i municipi

- token: jj6z-iyrp
- URL: https://analisi.transparenciacatalunya.cat/Salut/Registre-de-casos-de-COVID-19-realitzats-a-Catalun/jj6z-iyrp

### Registre de defuncions per COVID-19 a Catalunya. Segregació per sexe i comarca

- token: uqk7-bf9s
- URL: https://analisi.transparenciacatalunya.cat/Salut/Registre-de-defuncions-per-COVID-19-a-Catalunya-Se/uqk7-bf9s

***