import torch.cuda
import numpy as np
from torch.optim import Adam
from module.model import SPGAT
from util.parser import arg
from data_loader.CustomDataset import DataLoader
from tqdm import tqdm
from util.metrics import calc_metrics_at_k


device="cuda:0" if torch.cuda.is_available() else "cpu"

def train():
    data=DataLoader(arg)
    model=SPGAT(n_users=data.n_users,n_items=data.n_items,
                n_entities=data.n_entities,n_relations=data.n_relations,
                dim=arg.embed_dim,A=arg.A,drop_rate=arg.p,
                n_layers=arg.n_layers,L1_flag=arg.L1_flag,
                margin=arg.margin,M_social=data.social_graph,A_in=data.A_in).to(device)
    cf_optim = Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    kg_optim = Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    print(data.train_user_dict)
    # train model
    for epoch in range(1, arg.epoches + 1):
        model.train()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
        for iter in range(1, n_cf_batch + 1):
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict,data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            print(cf_batch_user)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_loss = model.clac_cf_loss(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
            cf_batch_loss.backward()
            cf_optim.step()
            cf_optim.zero_grad()
            cf_total_loss += cf_batch_loss.item()
            print("*********************")
            print(f"cf_loss:{cf_batch_loss.item()}")
            print("*********************")
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model.update_attention(h_list, t_list, r_list, relations)
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1
        kg_total_loss=0.0
        for iter in range(1, n_kg_batch + 1):
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail ,kg_batch_neg_rel= data.generate_kg_batch(
                data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)
            kg_batch_neg_rel  = kg_batch_neg_rel.to(device)
            kg_batch_loss = model.clac_kg_loss(kg_batch_head,kg_batch_pos_tail,kg_batch_relation,kg_batch_head,
                                               kg_batch_neg_tail,kg_batch_neg_rel)
            kg_batch_loss.backward()
            kg_optim.step()
            kg_optim.zero_grad()
            kg_total_loss += kg_batch_loss.item()
            print(kg_batch_loss.item())
        print("*********************")
        print(f"kg_loss:{kg_total_loss}")
        print("*********************")
    torch.save(model,"model.pkl")



def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model.calc_score(batch_user_ids, item_ids)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict



def predict(args):
    data = DataLoader(arg)
    model = torch.load("model.pkl")
    model.to(device)
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)
    cf_scores, metrics_dict = evaluate(model, data, Ks, device)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))





if __name__=='__main__':
    train()


