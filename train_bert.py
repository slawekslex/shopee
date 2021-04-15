from fastai.vision.all import *
import sklearn.metrics as skm
from tqdm.notebook import tqdm
import sklearn.feature_extraction.text
from transformers import (BertTokenizer, BertModel,AutoModel,
                          DistilBertTokenizer, DistilBertModel,AutoModelForSeq2SeqLM)

from shopee_utils import *
from train_utils import *
import argparse

BERT_MODEL_CLASS = AutoModel
BERT_TOKENIZER_CLASS = BertTokenizer
class CONF():
    bert_path ='indobenchmark/indobert-base-p2'
    arcface_m = .5
    arcface_s = 30
    lr = 1e-2
    lr_mult = 100
    train_epochs = 8
    train_freeze_epochs = 2
    do_mixup = True
    droput_p = .25
    embs_dim = 768
    tokens_max_length = 50
    adam_mom=.9
    adam_sqr_mom=.99
    adam_eps=1e-5
    adam_wd=0.01
    use_argmargin=True
    arc_easymargin=False
    label_smooth=.1
    experiment_id=0
    def toDict(self):
        return {k:self.__getattribute__(k) for k in dir(self) if k[:2]!='__' and not inspect.isroutine(self.__getattribute__(k))}
    
    def __repr__(self):
        return str(self.toDict())



class ArcFaceClassifier(nn.Module):
    def __init__(self, in_features, output_classes):
        super().__init__()
        self.initial_layers=nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(config.droput_p))
        self.W = nn.Parameter(torch.Tensor(in_features, output_classes))
        nn.init.kaiming_uniform_(self.W)
    def forward(self, x):
        x = self.initial_layers(x)
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        return x_norm @ W_norm
    
class ArcMarginProductLoss():
    def __init__(self,out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.out_features=out_features
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    def __call__(self, cosine, label):
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
    
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return F.cross_entropy(output, label)

class BertArcFace(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model
        self.classifier = ArcFaceClassifier(config.embs_dim, dls.c)
        self.outputEmbs = False
    def forward(self, x):
        output = self.bert_model(*x)
        last_hidden =output.last_hidden_state[:,0,:]
        embeddings=last_hidden
        if self.outputEmbs:
            return embeddings
        return self.classifier(embeddings)

class TitleTransform(Transform):
    def __init__(self):
        super().__init__()
        self.tokenizer = BERT_TOKENIZER_CLASS.from_pretrained(config.bert_path)
        
        
    def encodes(self, row):
        text = row.title
        encodings = self.tokenizer(text, padding = 'max_length', max_length=config.tokens_max_length,
                                   truncation=True,return_tensors='pt')
        keys =['input_ids', 'attention_mask']#, 'token_type_ids'] 
        return tuple(encodings[key].squeeze() for key in keys)

def new_model():
    bert_model = BERT_MODEL_CLASS.from_pretrained(config.bert_path)
    return BertArcFace(bert_model)

def split_2way(model):
    return L(params(model.bert_model),
            params(model.classifier))

######### MAIN
parser = argparse.ArgumentParser()
parser.add_argument("--bert_path", type=str)
parser.add_argument("--arcface_m", type=float)
parser.add_argument("--arcface_s", type=float)
parser.add_argument("--lr", type=float)
parser.add_argument("--lr_mult", type=float)
parser.add_argument("--train_epochs", type=int)
parser.add_argument("--train_freeze_epochs", type=int)
parser.add_argument("--droput_p", type=float)
parser.add_argument("--embs_dim", type=int)
parser.add_argument("--tokens_max_length", type=int)
parser.add_argument("--adam_mom", type=float)
parser.add_argument("--adam_sqr_mom", type=float)
parser.add_argument("--adam_eps", type=float)
parser.add_argument("--adam_wd", type=float)
parser.add_argument("--use_argmargin", type=int)
parser.add_argument("--arc_easymargin", type=int)
parser.add_argument("--label_smooth", type=float)
parser.add_argument("--experiment_id", type=int)
args = parser.parse_args()

config = ConfigClass.fromDict(vars(args))

print('Starting', config.experiment_id)
print(config)
train_df = pd.read_csv(PATH/'train_split.csv')
train_df['is_valid'] = train_df.split==0

tfm = TitleTransform()

data_block = DataBlock(
    blocks = (TransformBlock(type_tfms=tfm), 
              CategoryBlock(vocab=train_df.label_group.to_list())),
    splitter=ColSplitter(),
    get_y=ColReader('label_group'),
    )
dls = data_block.dataloaders(train_df, bs=128)

if config.use_argmargin:
    loss_func=ArcMarginProductLoss(dls.c, scale=config.arcface_s, margin=config.arcface_m, 
                               easy_margin=config.arc_easymargin,ls_eps=config.label_smooth)
else:
    loss_func=functools.partial(arcface_loss, m=config.arcface_m, s=config.arcface_s)
f1_tracker = TrackerCallback(monitor='F1 embeddings', comp=np.greater)
opt_func=functools.partial(Adam, mom=config.adam_mom, sqr_mom=config.adam_sqr_mom, eps=config.adam_eps, wd=config.adam_wd)
learn = Learner(dls,new_model(), opt_func=opt_func, splitter=split_2way, loss_func=loss_func,  
                cbs = [F1FromEmbs, f1_tracker], metrics=FakeMetric())

learn.fine_tune(config.train_epochs, config.lr, freeze_epochs=config.train_freeze_epochs, lr_mult=config.lr_mult)
print("SCORE", f1_tracker.best)

exp_res = pd.read_csv('shopee_results.csv')
row = {'id':config.experiment_id,  'hypers':str(config), 'f1_score':f1_tracker.best }
exp_res  =exp_res.append(row, ignore_index=True)
exp_res.to_csv('shopee_results.csv', index=False)