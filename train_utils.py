
import torch.nn.functional as F
from fastai.vision.all import *
from shopee_utils import *

def arcface_loss(cosine, targ, m=.5, s=30, output_classes=11014):
    cosine = cosine.clip(-1+1e-7, 1-1e-7) 
    arcosine = cosine.arccos()
    arcosine += F.one_hot(targ, num_classes = output_classes) * m
    cosine2 = arcosine.cos()
    cosine2 *= s
    return F.cross_entropy(cosine2, targ)



class F1FromEmbs(Callback):
    def after_pred(self):
        if not self.training:
            self.embs.append(self.learn.pred)
            self.ys.append(self.learn.yb[0])
            self.learn.yb = tuple()
    def before_validate(self):
        self.ys = []
        self.embs = []
        self.model.outputEmbs = True
    def before_train(self):
        self.model.outputEmbs = False
    def after_validate(self):
        embs = torch.cat(self.embs)
        embs = F.normalize(embs)
        ys = torch.cat(self.ys)
        score = f1_from_embs(embs,ys)
        self.learn.metrics[0].val = score

class FakeMetric(Metric):
    val =0.0
    @property
    def value(self):
        return self.val
    
    @property
    def name(self): 
        return 'F1 embeddings'


class ConfigClass():
    def toDict(self):
        return {k:self.__getattribute__(k) for k in dir(self) if k[:2]!='__' and not inspect.isroutine(self.__getattribute__(k))}
    
    def fromDict(d):
        res = ConfigClass()
        for k,v in d.items():
            res.__setattr__(k,v)
        return res

    def __repr__(self):
        return str(self.toDict())

