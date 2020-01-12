#   Artificial Intelligence for HPC Error Detection
This project presents an artificial intelligence algorithm to recognize abnormal events in the COSMA5 HPC system in Durham University.

##  Getting Started 

### Prerequisites
Download python3, the version used while implementing the project is pythonconda3/4.5.4 in COSMA system. 
> module load pythonconda3/4.5.4

### Usage
To run the project stages:
- execute pre-stage, use `python3 pre_stage/run.py`.
- execute stage1, use `python3 stage_1/run.py`. 
- execute stage2, use `python3 stage_1/run.py`.

Using the bash script `sh run` to run all the stages in COSMA
bash script will execute these commands:

```
module unload python/2.7.15
module load pythonconda3/4.5.4
python3 pre_stage/run.py
python3 stage_1/run.py 
python3 stage_2/run.py
```

