from fastai.vision.all import *
from tqdm.notebook import tqdm
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

def build_from_pairs(pairs, target, display = True):
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
    if display:
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

def hash_label(x):
    x = (13*x)%10000
    return x // 2000

def add_splits(data_df):
    data_df['split'] = data_df.label_group.apply(hash_label)
    data_df['is_valid'] = data_df.split == 0
    return data_df

def do_chunk(embs):
    step = 10000
    for chunk_start in range(0, embs.shape[0], step):
        chunk_end = min(chunk_start+step, len(embs))
        yield embs[chunk_start:chunk_end]

def embs_from_model(model, dl):
    all_embs = []
    all_ys=[]
    for batch in tqdm(dl):
        if len(batch) ==2:
            bx,by=batch
        else:
            bx,=batch
            by=torch.zeros(1)
        with torch.no_grad():
            embs = model(bx)
            all_embs.append(embs.half())
        all_ys.append(by)
    all_embs = F.normalize(torch.cat(all_embs))
    return all_embs, torch.cat(all_ys)

def f1_from_embs(embs, ys, display=False):
    target_matrix = ys[:,None]==ys[None,:]
    groups = [torch.where(t)[0].tolist() for t in target_matrix]
    dists, inds = get_nearest(embs, do_chunk(embs))
    pairs = sorted_pairs(dists, inds)[:len(embs)*10]
    scores =build_from_pairs(pairs, groups, display)
    return max(scores)
