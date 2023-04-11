import torch
import numpy as np
import random
from kmeans_pytorch import kmeans

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(input):
    torch.manual_seed(input)
    np.random.seed(input)
    random.seed(input)
    # print("In distributed training seed is : ", input)


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()


def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()


def majority_vote(target, sources, lr):
    for name in target:
        threshs = torch.stack([torch.max(source[name].data) for source in sources])
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()


# Score_Types = {accuracy,precision,recall,f1}
def score_averaging(target, clients, client_scores):
    sum_scores = sum([s for s in client_scores.values()])
    scores_per = {n: (s * 1.0) / sum_scores for n, s in client_scores.items()}

    sources = []
    for client in clients:
        source = {name: (value * scores_per[client.name]).to(device) for name, value in client.model.named_parameters()}
        sources.append(source)

    for name in target:
        target[name].data = torch.sum(torch.stack([source[name].data for source in sources]), dim=0).clone()


def score_averaging_cluster(target,clients_dict, client_scores):
    val = [s for s in client_scores.values()]
    flag = False
    for v in val:
        if v < np.mean(val) + 0.15 and v > np.mean(val) - 0.15:
            pass
        else:
            flag = True

    if flag:
        val = [[v] for v in val]
        val = torch.from_numpy(np.array(val)).to(device)
        kMeans = kmeans(X=val, num_clusters=2, device=device)

        cluster_1 = [float(val[index][0]) for index, i in enumerate(kMeans[0]) if int(i) == 0]
        cluster_2 = [float(val[index][0]) for index, i in enumerate(kMeans[0]) if int(i) == 1]

        selected = cluster_1 if np.mean(cluster_1) > np.mean(cluster_2) else cluster_2
    else:
        selected = val if np.mean(val) > 0.5 else []

    if selected == []:
        print("No Good Aggregation!!")
        return 0

    selected_client_scores = {n: s for n, s in client_scores.items() if s in selected}
    selected_client = {n: clients_dict[n] for n, s in client_scores.items() if s in selected}
    selected_client = [c for c in selected_client.values()]

    print("Number of selected:", len(selected_client))

    sum_scores = sum([s for s in selected_client_scores.values()])
    scores_per = {n: (s * 1.0) / sum_scores for n, s in selected_client_scores.items()}

    sources = []
    for client in selected_client:
        source = {name: (value.to(device) * scores_per[client.name]).to(device) for name, value in client.model.named_parameters()}
        sources.append(source)

    for name in target:
        target[name].data = target[name].data.to(device)

    for name in target:
        target[name].data = torch.sum(torch.stack([source[name].data for source in sources]), dim=0).clone()


def count_bits(T):
    k = T.numel()
    B_pos = 0
    b_total = (B_pos + 32) * k

    return b_total