import torch
import torch.nn as nn
import torch.nn.functional as F
import module.loss as loss
import torch.autograd as autograd

class Aggregator(nn.Module):
    def __init__(self,drop_rate,dim):
        super(Aggregator,self).__init__()
        self.drop_out=nn.Dropout(p=drop_rate)
        self.linear1=nn.Linear(dim,dim)
        self.linear2=nn.Linear(dim,1,bias=False)
        self.linear3=nn.Linear(dim,1,bias=False)
        self.linear4=nn.Linear(dim*2,dim)
        self.activation=F.leaky_relu
        self.activation1=F.tanh
    def aggregator_user_item_layers(self,all_emb,A_in):
        side_emb= torch.matmul(A_in, all_emb)
        return side_emb

    def _prepare_attentional_mechanism_input(self,user_emb):
        # dim=n_users * 1
        Wh1 = self.linear2(user_emb)
        # dim=n_user * 1
        Wh2 = self.linear3(user_emb)
        e = Wh1 + Wh2.T
        return self.activation(e)

    def aggregator_user_social_layers(self,user_emb,A_in):
        all_emb=user_emb
        e=self._prepare_attentional_mechanism_input(user_emb)
        zero_vector=torch.zeros(A_in.shape)
        attention=torch.where(A_in>0,e,zero_vector)
        # dim=n_user*n_user
        attention=F.softmax(attention,dim=1)
        attention=self.drop_out(attention)
        all_emb=torch.matmul(attention,all_emb)
        return all_emb


class SPGAT(nn.Module):

    def __init__(self,n_users,n_items,n_relations,n_entities,dim,A,drop_rate,n_layers,margin,L1_flag,M_social,A_in=None):
        super(SPGAT,self).__init__()
        self.entities_user_emb=nn.Embedding(n_users+n_entities,dim)
        self.rel_emb=nn.Embedding(n_relations,dim)

        self.entities_user_proj_emb = nn.Embedding(n_users+n_entities,dim)
        self.rel_proj_emb = nn.Embedding(n_relations,dim)
        self.entities_user_proj_emb.weight=nn.Parameter(torch.FloatTensor(n_users+n_entities,dim).zero_())
        self.rel_proj_emb.weight=nn.Parameter(torch.FloatTensor(n_relations,dim).zero_())

        self.criterion_cf=F.logsigmoid
        self.drop_out=nn.Dropout(p=drop_rate)
        self.n_users=n_users
        self.n_items=n_items
        self.n_entities=n_entities
        self.A=A
        self.aggregator=[Aggregator(drop_rate=drop_rate,dim=dim) for _ in range(n_layers)]
        # 用户社交网络
        self.M_social=M_social
        self.trans_M = nn.Parameter(torch.Tensor(n_relations,dim,dim))
        self.kg_loss_function= loss.marginLoss()
        self.margin=autograd.Variable(torch.FloatTensor([margin]).cuda())
        self.L1_flag=L1_flag
        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False
        self.linear1=nn.Linear(dim,dim,bias=False)
        self.linear2=nn.Linear(dim,dim,bias=False)
        self.activation=F.leaky_relu

    def bi_interaction(self,emb_1,emb_2):
        sum_embeddings = self.activation(self.linear1(emb_1 - emb_2))
        bi_embeddings = self.activation(self.linear2(emb_1 * emb_2))
        embeddings = bi_embeddings + sum_embeddings
        embeddings=self.drop_out(embeddings)
        return embeddings

    def clac_cf_loss(self,users,pos_item,neg_item):
        # users/pos_item/neg_item dim= batch_size * dim
        users_emb_social_final=None
        all_emb_cf_final=None

        users_emb_s=self.entities_user_emb.weight[self.n_entities:,]
        all_emb=self.entities_user_emb.weight

        for layers in self.aggregator:
            users_emb_social=layers.aggregator_user_social_layers(users_emb_s,self.M_social)
            all_emb_temp=layers.aggregator_user_item_layers(all_emb,self.A_in)

            users_emb_social=self.bi_interaction(users_emb_s,users_emb_social)
            all_emb=self.bi_interaction(all_emb,all_emb_temp)

            if users_emb_social_final==None:
                users_emb_social_final=users_emb_social
            else:
                users_emb_social_final=torch.cat([users_emb_social_final,users_emb_social],dim=1)

            if all_emb_cf_final==None:
                all_emb_cf_final=all_emb
            else:
                all_emb_cf_final=torch.cat([all_emb_cf_final,all_emb],dim=1)

        pos_emb_cf = all_emb_cf_final[pos_item]
        neg_emb_cf = all_emb_cf_final[neg_item]


        u_emb_social=users_emb_social_final[users-self.n_entities]
        user_pos=torch.sum(u_emb_social*pos_emb_cf,dim=1)
        user_neg=torch.sum(u_emb_social*neg_emb_cf,dim=1)
        cf_loss_social=-self.criterion_cf(user_pos-user_neg).mean()

        u_emb_cf = all_emb_cf_final[users]
        user_pos = torch.sum(u_emb_cf * pos_emb_cf, dim=1)
        user_neg = torch.sum(u_emb_cf * neg_emb_cf, dim=1)
        cf_loss_cf = -self.criterion_cf(user_pos - user_neg).mean()
        cf_loss=cf_loss_social+self.A*cf_loss_cf
        return cf_loss

    def projection_transD_pytorch_samesize(self,entity_embedding, entity_projection, relation_projection):
        return entity_embedding + torch.sum(entity_embedding * entity_projection, dim=1,
                                            keepdim=True) * relation_projection

    def clac_kg_loss(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.entities_user_emb(pos_h)
        pos_t_e = self.entities_user_emb(pos_t)
        pos_r_e = self.rel_emb(pos_r)
        pos_h_proj = self.entities_user_proj_emb(pos_h)
        pos_t_proj = self.entities_user_proj_emb(pos_t)
        pos_r_proj = self.rel_proj_emb(pos_r)

        neg_h_e = self.entities_user_emb(neg_h)
        neg_t_e = self.entities_user_emb(neg_t)
        neg_r_e = self.rel_emb(neg_r)
        neg_h_proj = self.entities_user_proj_emb(neg_h)
        neg_t_proj = self.entities_user_proj_emb(neg_t)
        neg_r_proj = self.rel_proj_emb(neg_r)

        pos_h_e = self.projection_transD_pytorch_samesize(pos_h_e, pos_h_proj, pos_r_proj)
        pos_t_e = self.projection_transD_pytorch_samesize(pos_t_e, pos_t_proj, pos_r_proj)
        neg_h_e = self.projection_transD_pytorch_samesize(neg_h_e, neg_h_proj, neg_r_proj)
        neg_t_e = self.projection_transD_pytorch_samesize(neg_t_e, neg_t_proj, neg_r_proj)

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)

        kg_loss = (self.kg_loss_function(pos, neg, self.margin)+loss.normLoss(self.entities_user_emb(torch.cat([pos_h, pos_t, neg_h, neg_t]))) +loss.normLoss(self.rel_emb(torch.cat([pos_r, neg_r])))
                   +loss.normLoss(pos_h_e) + loss.normLoss(pos_t_e) + loss.normLoss(neg_h_e) + loss.normLoss(neg_t_e))
        return kg_loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)
        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def calc_score(self, user_ids, item_ids):
        all_embed = self.calc_cf_embeddings()  # (n_users + n_entities, concat_dim)
        user_embed = all_embed[user_ids]  # (n_users, concat_dim)
        item_embed = all_embed[item_ids]  # (n_items, concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))  # (n_users, n_items)
        return cf_score









