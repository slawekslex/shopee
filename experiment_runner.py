
import subprocess
import random
from train_utils import *
import inspect


def draw_hypers():
    class CONF(ConfigClass):
        bert_path ='./indobert-base-p2'
        arcface_m = random.uniform(0.1,0.8)
        arcface_s = random.uniform(10,50)
        lr = random.uniform(1e-3,1e-2)
        lr_mult = random.randint(10,100)
        train_epochs = 12
        train_freeze_epochs = 1
        droput_p = random.uniform(0.1 ,0.5)
        embs_dim = 768
        tokens_max_length = random.randint(40,100)
        adam_mom=random.uniform(.8, .95)
        adam_sqr_mom=random.uniform(.95,.999)
        adam_eps=random.uniform(1e-6, 1e-4)
        adam_wd=random.uniform(0.001, 0.05)
        use_argmargin=random.choice([0,1])
        arc_easymargin=random.choice([0,1])
        label_smooth=random.uniform(0,0.2)
    return CONF().toDict()


for exp_id in range(100, 200):
    hypers = draw_hypers()   
    print(hypers)
    print('============= exp', exp_id, '\n', flush=True)
 
    com = ['python', '-W', 'ignore', 'train_bert.py', f'--experiment_id={exp_id}']
    for name,value in hypers.items():
        com.append(f'--{name}={value}')
    #com.append('--run_without_valid')
    subprocess.run(com)
    