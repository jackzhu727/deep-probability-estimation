# Introduction 
The atmosphere is governed by
nonlinear dynamics, hence weather forecast models possess inherent uncertainties (Richardson, 2007). Nowcasting,
weather prediction in the near future, is of great operational
significance, especially with increasing number of extreme
inclement weather conditions (Agrawal et al., 2019; Ravuri
et al., 2021). 
## Dataset
We use the [German Weather service dataset](https://opendata.dwd.de/weather/radar/), which contains quality-controlled rainfall-depth composites from 17 operational Doppler radars. 
We use 30 minutes of
precipitation data to predict if the mean precipitation over
the area covered will increase or decrease one hour after
the most recent measurement. Three precipitation maps
from the past 30 minutes serve as an input. The outcome
distribution is similar to the Linear scenario in Section 7.1
Three precipitation maps from the past 30 minutes serve as an input. 
The training labels are the 0/1 events indicating whether the mean precipitation increases (1) or not (0). 

The dataset contains quality-controlled rainfall-depth composites from 17 operational DWD Doppler radars. 
It has a spatial extent of 900x900 km, and covers the entirety of Germany. 
Data exists since 2006, with a spatial and temporal resolution of 1x1 km and 5 minutes, respectively. 
The dataset has been used to train RainNet, a pricipitation nowcasting model (Ayzel, G. Rainnet: a convolutional neural network for radarbased precipitation nowcasting. [RainNet](https://github.com/hydrogo/rainnet), 2020). 

We use a ResNet18 network architecture, with 3 input channels and 2 output channels.
The input to the network are 3 precipitation maps which cover a fixed area of 300km x 300 km in the center of the grid (300 x 300 pixels), set 10 minutes apart. The training, validation and test datasets consist of 20000, 6000 and 3000 samples, respectively, all separated temporally, over the span of 15 years.

## Dataloader
A pytorch implementation can be found [here](https://github.com/jackzhu727/deep-probability-estimation/blob/main/datasets/weather_forcast.py)

## Dataset
A preprocessed dataset used by the paper can be found [here]()
