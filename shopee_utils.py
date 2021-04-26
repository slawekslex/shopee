from fastai.vision.all import *
from tqdm.notebook import tqdm
import PIL
from sklearn.model_selection import StratifiedKFold

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

def score_group(group, target):
    tp = len(set(group).intersection(set(target)))
    return 2 * tp / (len(group)+len(target))
def score_all_groups(groups, targets):
    scores = [score_group(groups[i], targets[i]) for i in range(len(groups))]
    return sum(scores)/len(scores)

def build_from_pairs(pairs, target, display = True):
    score =0
    tp = [0]*len(target)
    fp = [0]*len(target)
    scores=[]
    vs=[]
    group_sizes = [len(x) for x in target]
    for x, y, v in pairs:
        group_size = group_sizes[x]
        score -= f1(tp[x], fp[x], group_size)
        if y in target[x]: tp[x] +=1
        else: fp[x] +=1
        score += f1(tp[x], fp[x], group_size) 
        scores.append(score / len(target))
        vs.append(v)
    if display:
        plt.plot(scores)
        am =torch.tensor(scores).argmax()
        print(f'{scores[am]:.3f} at {am/len(target)} pairs or {vs[am]:.3f} threshold')
    return scores



def sorted_pairs(distances, indices):
    triplets = []
    n= len(distances)
    for x in range(n):
        used=set()
        for ind, dist in zip(indices[x].tolist(), distances[x].tolist()):
            if not ind in used:
                triplets.append((x, ind, dist))
                used.add(ind)
    return sorted(triplets, key=lambda x: -x[2])

def get_nearest(embs, emb_chunks, K=None, sorted=True):
    if K is None:
        K = min(50, len(embs))
    distances = []
    indices = []
    for chunk in emb_chunks:
        sim = embs @ chunk.T
        top_vals, top_inds = sim.topk(K, dim=0, sorted=sorted)
        distances.append(top_vals.T)
        indices.append(top_inds.T)
    return torch.cat(distances), torch.cat(indices)

def get_dist_for_inds(embs, inds):
    step=1000
    dists = []
    for chunk_start in range(0, len(inds), step):
        chunk_end = min(chunk_start+step, len(inds))
        A = embs[chunk_start: chunk_end,:, None]
        B = embs[inds[chunk_start:chunk_end]]
        D = torch.matmul(B,A)
        dists.append(D.squeeze())
    dists = torch.cat(dists)
    return dists

def add_target_groups(data_df, source_column='label_group', target_column='target'):
    target_groups = data_df.groupby(source_column).indices
    data_df[target_column]=data_df[source_column].map(target_groups)
    return data_df



def add_splits(train_df, valid_group=0):
    grouped = train_df.groupby('label_group').size()

    labels, sizes =grouped.index.to_list(), grouped.to_list()

    skf = StratifiedKFold(5)
    splits = list(skf.split(labels, sizes))

    group_to_split =  dict()
    for idx in range(5):
        labs = np.array(labels)[splits[idx][1]]
        group_to_split.update(dict(zip(labs, [idx]*len(labs))))

    train_df['split'] = train_df.label_group.replace(group_to_split)
    train_df['is_valid'] = train_df['split'] == valid_group
    return train_df

def do_chunk(embs):
    step = 1000
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


from scipy.optimize import curve_fit
def objective(x, a, b, c):
	return a * np.exp(-b * x) + c

def do_scale(x,min_val,a,b,c):
    scaled = a*torch.exp(-b*x)+c
    return ((x>=min_val) * scaled).clamp(0,1)

def scale_sims(sims, target_matrix):
    scores, indices = sims.view(-1).topk(10*len(sims))
    width =1000
    target_flat = target_matrix.view(-1)
    probs,mean_sims=[],[]
    for bucket_start in range(0, len(scores), width):
        bucket_end = min(bucket_start+width, len(scores))
        bucket_inds = indices[bucket_start:bucket_end]
        targets = target_flat[bucket_inds]
        bucket_prob = targets.sum()/targets.numel()
        mean_sim = scores[bucket_start:bucket_end].mean()
        mean_sims.append(mean_sim.item())
        probs.append(bucket_prob.item())
    x, y= mean_sims, probs
    popt, _ = curve_fit(objective, x, y)

    # plot input vs output
    plt.scatter(x, y)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x), 1, 0.01)
    # calculate the output for the range
    y_line = objective(x_line, *popt)
    # create a line plot for the mapping function
    plt.plot(x_line, y_line, '--', color='red')
    plt.show()
    params = {'min_val':min(x), 'a':popt[0], 'b':popt[1], 'c':popt[2]}
    print(params)
    return functools.partial(do_scale,**params)


def get_targets_shape(train_df):
    all_targets = add_target_groups(train_df).target.to_list()
    all_targets_lens = [len(t) for t in all_targets]
    targets_shape = []
    for size in range(min(all_targets_lens), max(all_targets_lens)+1):
        count = all_targets_lens.count(size) / len(all_targets)
        targets_shape.append((size,count))
    return targets_shape

def cut(groups, groups_p, pos, target_count):
    probs = []
    groups_lens = [len(g)for g in groups]
    current_count = groups_lens.count(pos)
    if current_count >= target_count:

        return
    to_cut = target_count - current_count
    for i in range(len(groups)):
        if len(groups_p[i])>pos:
            probs.append((i, groups_p[i][pos]))
    probs.sort(key=lambda x:x[1])
    for i in range(min(to_cut, len(probs))):
        group_idx = probs[i][0] 
        groups[group_idx]=groups[group_idx][:pos]
        groups_p[group_idx]=groups_p[group_idx][:pos]
        
def group_and_shave(dists, combined_inds, train_df):
    triplets = sorted_pairs(dists, combined_inds)
    groups = [[] for _ in range(len(dists))]
    groups_p = [[] for _ in range(len(dists))]
    for x,y,v in triplets:
        if len(groups[x])>=51: continue
        groups[x].append(y)
        groups_p[x].append(v)
    targets_shape = get_targets_shape(train_df)
    for pos, size_pct in targets_shape:
        cut(groups, groups_p, pos, int(size_pct * len(groups)))
    return groups

def shave_and_score(D, targets, data):
    dists, inds = D.topk(50, dim=1)
    groups =group_and_shave(dists, inds, data)
    print(score_all_groups(groups, targets))


def gen_sim_and(embs1, embs2):
    res = torch.empty((len(embs1), len(embs1)), device = embs1.device, dtype=embs1.dtype)
    step = 500
    for chunk_start in range(0, embs1.shape[0], step):
        chunk_end = min(chunk_start+step, len(embs1))
        chunk1=embs1[chunk_start: chunk_end]
        chunk2=embs2[chunk_start: chunk_end]
        sim1 = chunk1@embs1.T
        sim2 = chunk2@embs2.T
        sim = sim1+sim2 - sim1*sim2
        res[chunk_start:chunk_end]=sim
    return res

def reciprocal_probs(D, x, tresh, scaled=False, include_x=False):
    neighb = torch.where(D[x]>tresh)
    if scaled:
        probs =D[x,neighb[0]]
        DP = probs[:,None] * D[neighb]
    else:
        DP = D[neighb]
    if include_x:
        DP = (DP.sum(dim=0) + D[x]) / (len(neighb[0])+1)
    else:
        DP = DP.mean(dim=0)
    return DP

def all_reciprocal(D, tresh, scaled=False, include_x=False):
    ret = [reciprocal_probs(D,i,tresh,scaled, include_x)[None] for i in range(len(D))]
    return torch.cat(ret)

def top_reciprocal(D, thresh, scaled=False, include_x=False):
    r_dists, r_ind=[],[]
    K = min(50, len(D))
    for x in range(len(D)):
        d_x = reciprocal_probs(D, x, thresh, scaled, include_x)
        v,i = d_x.topk(K)
        r_dists.append(v)
        r_ind.append(i)
    return r_dists, r_ind

def dist_to_pairs(dist):
    res = []
    for x in range(len(dist)):
        vals, ys = dist[x].topk(100)
        for v,y in zip(vals.tolist(),ys.tolist()):
            res.append((x,y,v))
    return sorted(res, key=lambda x: -x[2])

def score_distances(dist, targets, display=False):
    triplets = dist_to_pairs(dist)[:len(dist)*10]
    return max(build_from_pairs(triplets, targets, display))