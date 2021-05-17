
import subprocess
import random
from train_utils import *
import inspect
import pandas as pd

def draw_hypers(val_split=0):
    class CONF( ConfigClass):
        arcface_m = random.uniform(0.5,1.0)
        arcface_s = random.uniform(50,80)
        lr = random.uniform(0.004,0.008)
        lr_mult = random.randint(10,100)
        train_freeze_epochs = 1# random.choice([1,2])
        droput_p = random.uniform(0.0 ,0.25)
        linear_layers = random.choice([0,1])
        linear_width=random.randint(512,2048)
        bs = 80
        split_nfnet=1#random.choice([0,1])
        gradient_clip=random.choice([0,1])
        val_split=0
    conf = CONF()
    #conf.val_split=val_split
    return conf.toDict()


# for exp_id in range(234, 250):
#     hypers = draw_hypers(exp_id%5)   
#     print(hypers)
#     print('============= exp', exp_id, '\n', flush=True)
 
#     com = ['python', '-W', 'ignore', 'train_image.py', f'--experiment_id={exp_id}']
    
#     for name,value in hypers.items():
#         com.append(f'--{name}={value}')
#     print(com)
#     #com.append('--run_without_valid')
#     subprocess.run(com)
res_df = pd.read_csv('shopee_img_results.csv')
ids = [130, 144, 147, 151, 153]
for id in ids:
    hypers = ConfigClass.fromDict(eval(res_df.iloc[id].hypers))
    hypers.experiment_id +=300
    com = ['python', '-W', 'ignore', 'train_image.py']
        
    for name,value in hypers.toDict().items():
        com.append(f'--{name}={value}')
    print(com)
    #     #com.append('--run_without_valid')
    subprocess.run(com)
