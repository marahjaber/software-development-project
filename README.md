#   Artificial Intelligence for HPC Error Detection
This project presents an artificial intelligence algorithm to recognize abnormal events in the COSMA5 HPC system in Durham University.

##  Getting Started 

### Prerequisites
Download python3, the version used while implementing the project is pythonconda3/4.5.4 in COSMA system. 
> module load pythonconda3/4.5.4

### Usage

- Use training data folder `train_data`
- Use testing data folder `test_data`

- run the project:

Using the bash script `sh run` to run all the stages in COSMA

bash script will execute these commands:

```
module unload python/2.7.15
module load pythonconda3/4.5.4
cd pre_stage
python3 run.py
cd ../stage_1
python3 run.py
cd ../stage_2
python3 run.py
```

