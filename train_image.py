import sys
sys.path.append('../pytorch-image-models')
import timm
from fastai.vision.all import *
from shopee_utils import *
from train_utils import *
import sklearn.metrics as skm
from tqdm.notebook import tqdm

from fastai.vision.learner import _resnet_split
import h5py
import argparse

class CONF( ConfigClass):
    arcface_m = .5
    arcface_s = 30.0
    lr = 1e-2
    lr_mult = 100.0
    train_freeze_epochs = 1
    droput_p = .25
    linear_layers = 3
    linear_width = 2048
    bs = 80
    split_nfnet=0
    gradient_clip=1
    experiment_id=666
    val_split=8
OUTPUT_CLASSES=11014
def_config=CONF()

def eca_nfnet_l0(pretrained): return timm.create_model("eca_nfnet_l0", pretrained = pretrained)
def eca_nfnet_l1(pretrained): return timm.create_model("eca_nfnet_l1", pretrained = pretrained)
arch = eca_nfnet_l1


def get_img_file(row):
    img =row.image
    fn  = PATH/'train_images'/img
    if not fn.is_file():
        fn = PATH/'test_images'/img
    return fn



def get_dls(size, bs):
    data_block = DataBlock(blocks = (ImageBlock(), CategoryBlock(vocab=train_df.label_group.to_list())),
                 splitter=ColSplitter(),
                 get_y=ColReader('label_group'),
                 get_x=get_img_file,
                 item_tfms=Resize(size*2, resamples=(Image.BICUBIC,Image.BICUBIC)),
                 batch_tfms=aug_transforms(size=size, min_scale=0.75)+[Normalize.from_stats(*imagenet_stats)],
                 )
    return data_block.dataloaders(train_df, bs=bs)

def save_without_classifier(model, fname):
    model.classifier = None
    torch.save(model.state_dict(), fname)

class ArcFaceClassifier(nn.Module):
    def __init__(self, in_features, output_classes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(in_features, output_classes))
        nn.init.kaiming_uniform_(self.W)
    def forward(self, x):
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        return x_norm @ W_norm
    


class ResnetArcFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = create_body(arch, cut=-2)
        nf = num_features_model(nn.Sequential(*self.body.children()))
        after_conv=[
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(nf*2),
            nn.Dropout(conf.droput_p)
        ]
        for i in range(conf.linear_layers):
            f = nf*2 if i==0 else 512
            after_conv.append(nn.ReLU())
            after_conv.append(nn.Linear(f,conf.linear_width))
            after_conv.append(nn.BatchNorm1d(conf.linear_width))
            after_conv.append(nn.Dropout(conf.droput_p))
        self.after_conv = nn.Sequential(*after_conv)    
        self.classifier = ArcFaceClassifier(conf.linear_width if conf.linear_layers>0 else nf*2, OUTPUT_CLASSES)
        self.outputEmbs = False
    def forward(self, x):
        x = self.body(x)
        embeddings = self.after_conv(x)
        if self.outputEmbs:
            return embeddings
        return self.classifier(embeddings)

def split_2way(model):
    return L(params(model.body) + params(model.after_conv),
            params(model.classifier))

def modules_params(modules):
    return list(itertools.chain(*modules.map(params)))

def split_nfnet(model):
    body =model.body 
    children = L(body.children())
    group1 =children[:1]
    group2 = children[1:]
    group3 = L([model.after_conv,model.classifier])
    return [modules_params(g) for g in [group1,group2,group3]]



# ----------------------------------------------------------
parser = argparse.ArgumentParser()
for k,v in def_config.toDict().items():
    parser.add_argument('--'+k, type=type(v))
args = parser.parse_args()
conf = ConfigClass.fromDict(vars(args))

print('Starting', conf.experiment_id)
print(conf)
train_df = add_splits(pd.read_csv(PATH/'train.csv'),conf.val_split )
valid_df = train_df[train_df.is_valid==True].copy()

train_df.is_valid=False

train_df= pd.concat([train_df, valid_df])
#opt_func=RMSProp
opt_func=Adam
if conf.split_nfnet:
    split_func= split_nfnet
else:
    split_func = split_2way
loss_func=functools.partial(arcface_loss, m=conf.arcface_m, s=conf.arcface_s)
f1_tracker = TrackerCallback(monitor='F1 embeddings', comp=np.greater)
learn = Learner(get_dls(224, 96),ResnetArcFace(),splitter=split_func, 
                opt_func=opt_func, loss_func=arcface_loss, cbs = [F1FromEmbs,f1_tracker], metrics=FakeMetric()).to_fp16()
if conf.gradient_clip:
    learn.cbs.append(GradientClip)

learn.fine_tune(10,conf.lr, freeze_epochs = conf.train_freeze_epochs, lr_mult=conf.lr_mult)

learn.save('stage1')


learn = Learner(get_dls(336, 48), ResnetArcFace(),splitter=split_func, 
                opt_func=opt_func, loss_func=arcface_loss, cbs = [F1FromEmbs,f1_tracker], metrics=FakeMetric()).to_fp16()
learn.load('stage1')
print('----upscale')
learn.fine_tune(6, 1e-4)
print('saving',f'models/nfnetl0_336_noval{conf.experiment_id}.pth' )
save_without_classifier(learn.model, f'models/nfnetl0_336_noval{conf.experiment_id}.pth')