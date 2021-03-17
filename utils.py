from fastai.vision.all import *
import PIL
PATH = Path('/home/slex/data/shopee')

def show(data):
    n = min(len(data), 40)
    _,axs = plt.subplots((n+1)//2,2, figsize=(20,2*n))
    for ax, (_,row) in zip(axs.flatten(), data.iterrows()):
        img_path = PATH/'train_images'/row.image
        ax.imshow(PIL.Image.open(img_path))
        ax.axis('off')
        txt = f'{row["title"]}'
        ax.set_title(txt) 