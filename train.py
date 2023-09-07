# %%
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning .callbacks import ModelCheckpoint
import torch
import yaml
import os.path as Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from VAE import VAEAnomalyTabular
from dataset import rand_dataset

import pandas as pd

# %%
class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index].values

# %%
ROOT = Path.dirname(__file__)
SAVED_MODELS = Path.join(ROOT, 'saved_models')

# %%
input_size = 7
latent_size = 5
num_resamples = 10
epochs = 10
batch_size = 32
device = 'cpu'
lr = 1e-3
no_progress_bar = True
steps_log_loss = 1000
steps_log_norm_params = 1000

# %%
model = VAEAnomalyTabular(input_size, latent_size, num_resamples, lr)

# %%
# # data
df = pd.read_csv("TravelTime_387.csv", parse_dates=['timestamp'])
df_hourly = df.set_index('timestamp').resample('H').mean().reset_index()
df_daily = df.set_index('timestamp').resample('D').mean().reset_index()
# New features
# Loop to cycle through both DataFrames
for DataFrame in [df_hourly, df_daily]:
    DataFrame['Weekday'] = (pd.Categorical(DataFrame['timestamp'].dt.strftime('%A'),
                                           categories=['Monday','Tuesday','Wednesday',
                                                       'Thursday','Friday','Saturday','Sunday'])) 
    DataFrame['Hour'] = DataFrame['timestamp'].dt.hour
    DataFrame['Day'] = DataFrame['timestamp'].dt.weekday
    DataFrame['Month'] = DataFrame['timestamp'].dt.month
    DataFrame['Year'] = DataFrame['timestamp'].dt.year
    DataFrame['Month_day'] = DataFrame['timestamp'].dt.day
    DataFrame['Lag'] = DataFrame['value'].shift(1)
    DataFrame['Rolling_Mean'] = DataFrame['value'].rolling(7, min_periods=1).mean()
    DataFrame = DataFrame.dropna()
    
df_hourly = (df_hourly
             .join(df_hourly.groupby(['Hour','Weekday'])['value'].mean(),
                   on = ['Hour', 'Weekday'], rsuffix='_Average')
            )
df_hourly.dropna(inplace=True)
model_data0 = df_hourly[['value', 'Hour', 'Day', 'Month_day', 'Month','Rolling_Mean','Lag', 'timestamp']].set_index('timestamp').dropna()
#normalize columns
model_data = (model_data0-model_data0.mean())/model_data0.std()
model_data = model_data.astype('float32')
# %%
num_samples = model_data.shape[0]
train = model_data[:int(num_samples*0.7)]
val = model_data[int(num_samples*0.7):]
# %%
train_set = PandasDataset(train)
val_set = PandasDataset(val)
train_dloader = DataLoader(train_set, batch_size, num_workers=4)
val_dloader = DataLoader(val_set, batch_size, num_workers=4)

# %%
# train_set = rand_dataset(num_columns=input_size)  
# train_dloader = DataLoader(train_set, batch_size)

# val_dataset = rand_dataset(num_columns=input_size)  
# val_dloader = DataLoader(val_dataset, batch_size)
# %%
checkpoint = ModelCheckpoint(
    dirpath=SAVED_MODELS,
    # filename='{epoch:02d}-{val_loss:.2f}',
    filename='traffic-{val_loss:.2f}',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    save_last=True,
)

trainer = Trainer(callbacks=[checkpoint],)
trainer.fit(model, train_dloader, val_dloader)
