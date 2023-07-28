import torch
from torch import nn
    
    
class BanzhafModule(nn.Module):
    def __init__(self, planes=512):
        super(BanzhafModule, self).__init__()
        self.cnn1 = nn.Conv2d(1, planes, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.cnn2 = nn.Conv2d(planes, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.Q = nn.Parameter(torch.empty(size=(planes, planes)))
        nn.init.xavier_uniform_(self.Q.data, gain=1.414)
        self.K = nn.Parameter(torch.empty(size=(planes, planes)))
        nn.init.xavier_uniform_(self.K.data, gain=1.414)
        self.V = nn.Parameter(torch.empty(size=(planes, planes)))
        nn.init.xavier_uniform_(self.V.data, gain=1.414)

    def forward(self, x):
        x = x.contiguous()
        b, a, t, v = x.size()
        x = x.view(b*a, 1, t, v)
        x = self.relu(self.cnn1(x))

        _x = x.permute(0, 2, 3, 1).view(b*a, t*v, -1)
        _q = torch.einsum('bld,dn->bln', [_x, self.Q])
        _k = torch.einsum('bld,dn->bln', [_x, self.K])
        _v = torch.einsum('bld,dn->bln', [_x, self.V])

        attn = torch.einsum('bqn,bkn->bqk', [_q, _k])
        attn = torch.softmax(attn, dim=-1)
        _x = torch.einsum('bqk,bkn->bqn', [attn, _v])

        _x = _x.view(b*a, t, v, -1).permute(0, 3, 1, 2)
        x = x + _x
        x = self.cnn2(x)
        x = x.view(b, a, t, v)
        return x

@torch.no_grad()
def banzhaf_values(retrieve_logits, text_mask, video_mask, text_weight, video_weight):
    text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float(-9e15))
    text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t
    video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool), float(-9e15))
    video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

    retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
    retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])

    t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
    t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

    v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
    v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
    banzhaf_values = (t2v_logits + v2t_logits) / 2.0
    return banzhaf_values


class BanzhafInteraction:
    def __init__(self, t_len, v_len, num=1000):
        self.t_len = t_len
        self.v_len = v_len
        self.num = num

    @torch.no_grad()
    def __call__(self, retrieve_logits, text_mask, video_mask, text_weight, video_weight):
        banzhaf = torch.zeros_like(retrieve_logits).to(retrieve_logits.device)
        retrieve_logits = retrieve_logits * 1000
        for i in range(self.t_len):
            for j in range(self.v_len):
                for _ in range(self.num):
                    banzhaf[:, i, j] = self.banzhaf_interaction(retrieve_logits, text_mask, video_mask, text_weight,
                                                                    video_weight, i, j)
        banzhaf = banzhaf / self.num
        banzhaf = torch.einsum('btv,bt->btv', [banzhaf, text_mask])
        banzhaf = torch.einsum('btv,bv->btv', [banzhaf, video_mask])
        return banzhaf

    @torch.no_grad()
    def banzhaf_interaction(self, retrieve_logits, text_mask, video_mask, text_weight, video_weight,
                        i, j):
        s_t = (torch.rand(text_mask.size()) > 0.5).long().to(retrieve_logits.device)
        s_j = (torch.rand(video_mask.size()) > 0.5).long().to(retrieve_logits.device)
        s_t[:, i], s_j[:, j] = 0, 0

        _text_mask, _video_mask = text_mask.clone(), video_mask.clone()
        _text_mask.masked_fill_((1 - s_t).to(torch.bool), 0)
        _video_mask.masked_fill_((1 - s_j).to(torch.bool), 0)

        #########################
        _text_weight0, _video_weight0 = text_weight.clone(), video_weight.clone()
        _retrieve_logits0 = retrieve_logits.clone()

        _retrieve_logits0 = torch.einsum('btv,bt->btv', [_retrieve_logits0, _text_mask])
        _retrieve_logits0 = torch.einsum('btv,bv->btv', [_retrieve_logits0, _video_mask])

        _text_weight0.masked_fill_((1 - _text_mask).to(torch.bool), float(-9e15))
        _text_weight0 = torch.softmax(_text_weight0, dim=-1)  # B x N_t
        _video_weight0.masked_fill_((1 - _video_mask).to(torch.bool), float(-9e15))
        _video_weight0 = torch.softmax(_video_weight0, dim=-1)  # B x N_v

        t2v_logits, max_idx1 = _retrieve_logits0.max(dim=-1)  # btv -> bt
        t2v_logits = torch.einsum('bt,bt->b', [t2v_logits, _text_weight0])

        v2t_logits, max_idx2 = _retrieve_logits0.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('bv,bv->b', [v2t_logits, _video_weight0])
        banzhaf_value4 = (t2v_logits + v2t_logits) / 2.0

        #########################
        _retrieve_logits0[:, i, j] = retrieve_logits[:, i, j]

        _text_weight0.masked_fill_((1 - _text_mask).to(torch.bool), float(-9e15))
        _text_weight0[:, i] = text_weight[:, i]
        _text_weight0 = torch.softmax(_text_weight0, dim=-1)  # B x N_t
        _video_weight0.masked_fill_((1 - _video_mask).to(torch.bool), float(-9e15))
        _video_weight0[:, j] = video_weight[:, j]
        _video_weight0 = torch.softmax(_video_weight0, dim=-1)  # B x N_v

        t2v_logits, max_idx1 = _retrieve_logits0.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('bt,bt->b', [t2v_logits, _text_weight0])

        v2t_logits, max_idx2 = _retrieve_logits0.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('bv,bv->b', [v2t_logits, _video_weight0])
        banzhaf_value1 = (t2v_logits + v2t_logits) / 2.0

        #########################
        _text_mask[:, i] = 1

        _text_weight0, _video_weight0 = text_weight.clone(), video_weight.clone()
        _retrieve_logits0 = retrieve_logits.clone()

        _retrieve_logits0 = torch.einsum('btv,bt->btv', [_retrieve_logits0, _text_mask])
        _retrieve_logits0 = torch.einsum('btv,bv->btv', [_retrieve_logits0, _video_mask])

        _text_weight0.masked_fill_((1 - _text_mask).to(torch.bool), float(-9e15))
        _text_weight0 = torch.softmax(_text_weight0, dim=-1)  # B x N_t
        _video_weight0.masked_fill_((1 - _video_mask).to(torch.bool), float(-9e15))
        _video_weight0 = torch.softmax(_video_weight0, dim=-1)  # B x N_v

        t2v_logits, max_idx1 = _retrieve_logits0.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('bt,bt->b', [t2v_logits, _text_weight0])

        v2t_logits, max_idx2 = _retrieve_logits0.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('bv,bv->b', [v2t_logits, _video_weight0])
        banzhaf_value2 = (t2v_logits + v2t_logits) / 2.0

        #########################
        _video_mask[:, j] = 1

        _text_weight0, _video_weight0 = text_weight.clone(), video_weight.clone()
        _retrieve_logits0 = retrieve_logits.clone()

        _retrieve_logits0 = torch.einsum('btv,bt->btv', [_retrieve_logits0, _text_mask])
        _retrieve_logits0 = torch.einsum('btv,bv->btv', [_retrieve_logits0, _video_mask])

        _text_weight0.masked_fill_((1 - _text_mask).to(torch.bool), float(-9e15))
        _text_weight0 = torch.softmax(_text_weight0, dim=-1)  # B x N_t
        _video_weight0.masked_fill_((1 - _video_mask).to(torch.bool), float(-9e15))
        _video_weight0 = torch.softmax(_video_weight0, dim=-1)  # B x N_v

        t2v_logits, max_idx1 = _retrieve_logits0.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('bt,bt->b', [t2v_logits, _text_weight0])

        v2t_logits, max_idx2 = _retrieve_logits0.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('bv,bv->b', [v2t_logits, _video_weight0])
        banzhaf_value3 = (t2v_logits + v2t_logits) / 2.0

        return banzhaf_value1 - banzhaf_value2 - banzhaf_value3 + banzhaf_value4
