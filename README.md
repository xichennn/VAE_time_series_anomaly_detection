# Variational autoencoder for anomaly detection in time-series data

- The dataset I used here is the Real time traffic data from the Twin Cities Metro area in Minnesota, which can be found [here](https://github.com/numenta/NAB/tree/master/data)
- The file names are quite self-explanatory to run the model
- Model results when using a probability threshold 0.2:
<p align="center">
  <img src="https://github.com/xichennn/VAE_time_series_anomaly_detection/blob/main/figs/trainset.png" width="350" title="q1">
  <img src="https://github.com/xichennn/VAE_time_series_anomaly_detection/blob/main/figs/valset.png" width="350" title="q2">
</p>

- TODO: replace MLP encoder decoder to LSTM encoder decoder
