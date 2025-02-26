# FedPP
This repository is the official implementation of the Federated Neural Nonparametric Point Processes.
## Install the Requirments of Experiment
```
pip install scipy
pip install tqdm
pip install matplotlib
pip install torch
```

## Running
### Dataset Selection
We choose five benchmark datasets to evaluate our method: 

(1) Taobao, which is a public dataset generated for the 2018 Tianchi Big Data Competition. It comprises timestamped behavioral records (such as browsing and purchasing activities) of anonymized users on the online shopping platform Taobao. The data spans from November 25 to December 03, 2017. The dataset comprises a total of 5,318 sequences and K = 17 types of events. 

(2) Retweet, which contains a total of 24,000 retweet sequences, where each sequence is composed of events represented as tuples, indicating tweet types and their corresponding times. There are K = 3 distinct types of retweeters: small, medium, and large. To classify retweeters into these categories, small retweeters have fewer than 120 followers, medium retweeters possess more than 120 but fewer than 1,363 followers, and the remaining retweeters are labeled as large. The dataset provides information on when a post will be retweeted and by which type of user. 

(3) Conttime, which is a public dataset releases by Mei & Eisner (2017). There are 9,000 sequences with a total of K = 5 event types. 

(4) Stack Overflow, which is a public dataset that encompasses sequences of user awards spanning a two-year period. In the Stack Overflow question-answering platform, recognizes users through awards based on their contributions, including posing insightful questions and providing valuable answers. The dataset comprises a total of 6,633 sequences and K = 22 types of events. 

(5) Amazon, which is a public dataset similar to Taobao, containing a timestamped behavioral record of anonymized users on the online shopping platform Amazon. There are 14,759 sequences with a total of K = 16 event types.



### Training
Running FedPP at different dataset
```
python main.py --dataset_type Taobao 
```
where dataset_type can be selected from amazon, conttime, retweet, stackoverflow, and taobao. 

Running FedPP with different aggregation schemes.
```
python main.py --aggerate_method FedEvent
```
where aggerate_method can be selected from FedAvg, AggSigma, AggSigma2 and FedEvent. 
