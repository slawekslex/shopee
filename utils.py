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


def f1(tp, fp, num_tar):
    return 2 * tp / (tp+fp+num_tar)

def build_from_pairs(pairs, target):
    score =0
    tp = [0]*len(target)
    fp = [0]*len(target)
    scores=[]
    group_sizes = [len(x) for x in target]
    for x, y, v in pairs:
        group_size = group_sizes[x]
        score -= f1(tp[x], fp[x], group_size)
        if y in target[x]: tp[x] +=1
        else: fp[x] +=1
        score += f1(tp[x], fp[x], group_size) 
        scores.append(score / len(target))
    plt.plot(scores)
    am =torch.tensor(scores).argmax()
    print(f'{scores[am]:.3f} at {am/len(target)} pairs')
    return scores


def sorted_pairs(distances, indices):
    triplets = []
    n, m = distances.shape
    for x in range(n):
        tri = zip([x] * m, indices[x].tolist(), distances[x].tolist())
        triplets += list(tri)
    
    return sorted(triplets, key=lambda x: -x[2])

def get_nearest(embs, emb_chunks):
    K = min(50, len(embs))
    distances = []
    indices = []
    for chunk in emb_chunks:
        sim = embs @ chunk.T
        top_vals, top_inds = sim.topk(K, dim=0)
        distances.append(top_vals.T)
        indices.append(top_inds.T)
    return torch.cat(distances), torch.cat(indices)

def add_target_groups(data_df):
    target_groups = data_df.groupby('label_group').indices
    data_df['target']=data_df.label_group.map(target_groups)
    return data_df
