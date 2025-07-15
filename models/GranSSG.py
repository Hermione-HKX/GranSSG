"""
Author: HKX
Date: January 2024
"""

import einops
import numpy as np
import pointops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def embedding_expand(emb, off):
    """ expanding the embedding according to the offset """
    assert emb.shape[0] == len(off)
    emb_list = []
    for o in range(len(off)):
        o_ = off[o] - off[o - 1] if o > 0 else off[0]
        e = emb[o].reshape(1, -1).repeat(o_, 1)
        emb_list.append(e)
    return torch.cat(emb_list, dim=0)


class LayerNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()


def index_points_2(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx.long(), :]
    return new_points


def index_points(points, idx):
    """
    Input:
        points: input points data, [N, C]
        idx: sample index data, [S]
    Return:
        new_points:, indexed points data, [S, C]
    """
    new_points = points[idx.long(), :]
    return new_points


def saved_fps(p, features, o, on, ins_mark=None):
    """
    For each instance in a scene, at least save one points. Then use the stander FPS
    the ins_mask [[a1, b1, c1, ...], [a2, b2, c2, ...]] a batch of list that contains the instances ids
    """
    ins_len, ins_idx = [], []
    new_on = []
    for batch in range(len(o)):
        batch_ins_mark = ins_mark[batch]
        o_start = o[batch - 1] if batch > 0 else 0
        o_end = o[batch]
        f_b = features[o_start:o_end]
        b_idx = [torch.where(f_b[:, -1] == x)[0][0] + o_start for x in batch_ins_mark]  # find the first idx from ins
        ins_len.append(len(batch_ins_mark))
        ins_idx.append(b_idx)
        # avoid instance len > offset
        on_start = on[batch - 1] if batch > 0 else 0
        on_end = on[batch]
        if on_end - on_start < len(batch_ins_mark):
            new_on.append(on_start + 1)
        else:
            new_s = new_on[-1] if batch > 0 else 0
            new_on.append(new_s + on_end - on_start - len(batch_ins_mark))
    new_on = torch.tensor(new_on).to(p.device)
    idx = pointops.farthest_point_sampling(p.contiguous(), o, new_on)  # (m)
    # adding saved point's index; rebuild new_offset with saved points
    new_idx = []
    return_no = []
    for batch in range(len(new_on)):
        on_start = new_on[batch - 1] if batch > 0 else 0
        on_end = new_on[batch]
        b_idx = idx[on_start:on_end]
        add_idx = torch.cat((b_idx, torch.tensor(ins_idx[batch]).to(p.device)))
        new_idx.append(add_idx)
        if batch == 0:
            return_no.append(on_end - on_start + len(ins_idx[batch]))
        else:
            return_no.append(return_no[-1] + on_end - on_start + len(ins_idx[batch]))
    # === assert ===
    for b in range(len(ins_mark)):
        S_FPS_mask = features[new_idx[b]][:, -1].unique()
        mask_ins = S_FPS_mask[1:] if -1 in S_FPS_mask else S_FPS_mask
        assert (mask_ins == ins_mark[b].unique()).sum() == ins_mark[b].shape[0]
    return torch.cat(new_idx), torch.tensor(return_no).to(p.device)


def filter_candidate_pairs(subject_ins, object_ins, tau_rate=0.5, min_edges=50):
    """
    Filter subject-object pairs based on cosine similarity.

    Args:
        subject_ins: [E, D] tensor of subject features
        object_ins: [E, D] tensor of object features
        tau: similarity threshold

    Returns:
        keep_edge_idx: [E] keep the edges or not --> bool
    """
    min_topk_edges = min(subject_ins.size(0), min_edges)
    # Normalize features to compute cosine similarity
    subj_norm = F.normalize(subject_ins, dim=1)
    obj_norm = F.normalize(object_ins, dim=1)

    # Compute cosine similarity for each pair: [E]
    sim_scores = F.cosine_similarity(subj_norm, obj_norm, dim=1)
    tau = sim_scores.mean() + tau_rate * sim_scores.std()

    # Threshold filtering
    keep_mask = sim_scores > tau

    # If not enough edges are kept, select top-min_edges highest similarities
    if keep_mask.sum() < min_topk_edges:
        topk_vals, topk_indices = torch.topk(sim_scores, min_topk_edges)
        keep_mask = torch.zeros_like(sim_scores, dtype=torch.bool)
        keep_mask[topk_indices] = True

    return keep_mask


class TransitionUp(nn.Module):
    def __init__(self, input_planes):
        """ build up-sampling layers """
        super().__init__()
        self.input_planes = input_planes
        upper_planes = input_planes * 2
        self.linear_cross = nn.Sequential(nn.Linear(upper_planes, input_planes), nn.BatchNorm1d(input_planes),
                                          nn.ReLU())
        self.linear_out = nn.Sequential(nn.Linear(input_planes, input_planes), nn.BatchNorm1d(input_planes),
                                        nn.ReLU())

    def forward(self, pxo1, pxo2):
        p1, x1, o1 = pxo1
        p2, x2, o2 = pxo2

        x = x1 + self.linear_cross(pointops.interpolation(p2, p1, x2, o2, o1))
        x = self.linear_out(x)
        return x


class Up_sampling(nn.Module):
    def __init__(self, input_dims):
        """ build up-sampling layers """
        super().__init__()
        self.up4 = TransitionUp(input_dims[0])  # 256
        self.up3 = TransitionUp(input_dims[1])  # 128
        self.up2 = TransitionUp(input_dims[2])  # 64
        self.up1 = TransitionUp(input_dims[3])  # 32

    def forward(self, pxo):
        p_list, x_list, o_list = pxo

        # upper layer,  lower layer
        up = 'near'
        if up == 'full':
            x4 = self.up4([p_list[3], x_list[3], o_list[3]], [p_list[4], x_list[4], o_list[4]])
            x3 = self.up3([p_list[2], x_list[2], o_list[2]], [p_list[3], x4, o_list[3]])
            x2 = self.up2([p_list[1], x_list[1], o_list[1]], [p_list[2], x3, o_list[2]])
            x1 = self.up1([p_list[0], x_list[0], o_list[0]], [p_list[1], x2, o_list[1]])
        else:
            x4 = self.up4([p_list[3], x_list[3], o_list[3]], [p_list[4], x_list[4], o_list[4]])
            x3 = self.up3([p_list[2], x_list[2], o_list[2]], [p_list[3], x_list[3], o_list[3]])
            x2 = self.up2([p_list[1], x_list[1], o_list[1]], [p_list[2], x_list[2], o_list[2]])
            x1 = self.up1([p_list[0], x_list[0], o_list[0]], [p_list[1], x_list[1], o_list[1]])
        pyramid_x = [x1, x2, x3, x4, x_list[4]]   # 32, 64, 128, 256, 512
        return pyramid_x


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3),
                                      LayerNorm1d(3),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(LayerNorm1d(mid_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, out_planes // share_planes),
                                      LayerNorm1d(out_planes // share_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxof) -> torch.Tensor:
        p, x, o, f = pxof  # (n, 3), (n, c), (b), (n, 11)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = pointops.knn_query_and_group(x_k, p, o, new_xyz=p, new_offset=o,
                                                nsample=self.nsample, with_xyz=True)
        x_v, _ = pointops.knn_query_and_group(x_v, p, o, new_xyz=p, new_offset=o,
                                              idx=idx, nsample=self.nsample, with_xyz=False)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]

        p_r = self.linear_p(p_r)
        r_qk = x_k - x_q.unsqueeze(1) + einops.reduce(p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes)
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum("n t s i, n t i -> n s i",
                         einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes), w)
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x


class Pyramid_Transformer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8):
        super().__init__()
        self.mid_planes = mid_planes = out_planes
        self.out_planes = out_planes
        self.share_planes = share_planes

        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, mid_planes)

        self.linear_w = nn.Sequential(LayerNorm1d(mid_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, out_planes // share_planes),
                                      LayerNorm1d(out_planes // share_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, emb, bbox) -> torch.Tensor:
        # x_q, x_k, x_v = self.linear_q(x + emb[:, None, :]), self.linear_k(x + emb[:, None, :]), x
        x_k, x_v = self.linear_k(x), self.linear_v(x)

        p_r = emb
        r_qk = x_k + emb[:, None, :]
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum("n t s i, n t i -> n s i",
                         einops.rearrange(x_v + p_r[:, None, :], "n ns (s i) -> n ns s i", s=self.share_planes), w)
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x


class Matrix_Transformer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)

        self.bn = nn.BatchNorm1d(dim)

    def forward(self, subj, obj, s_o_dot, subj_emb, obj_emb, delta_emb):
        use_bn = True
        use_norm = False

        q = self.linear_q(subj + subj_emb.repeat(1, 2)[:, None, :])
        k = self.linear_k(obj + obj_emb.repeat(1, 2)[:, None, :])
        if use_bn:
            q, k = self.bn(q.permute(0, 2, 1)).permute(0, 2, 1), self.bn(k.permute(0, 2, 1)).permute(0, 2, 1)

        v = self.linear_v(s_o_dot + delta_emb.repeat(1, 2)[:, None, :])

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2))
        dots = dots.view(dots.shape[0], dots.shape[1], -1) * self.scale

        if use_norm:
            dots_min, dots_max = torch.min(dots, dim=-1), torch.max(dots, dim=-1)
            dots = (dots - dots_min[0][:, :, None]) / (dots_max[0] - dots_min[0])[:, :, None]

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # attn = attn.view(attn.shape[0], attn.shape[1], -1) / subj.shape[1]  # obj, heads, att_weight (4*4)
        out = attn[:, :, :, None] * v   # obj, heads, att-channels, dim/heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = torch.sum(out, dim=1)
        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)

        # in_planes, out_planes, heads, gather_nei, nsample, drop=0.0, bn=True
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxodf):
        p, x, o, d, f = pxodf  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o, f])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o, d, f]


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16, ds='S-FPS'):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        self.down_sampling = ds
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxofd):
        p, x, o, f, down_mask = pxofd  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            if self.down_sampling == 'FPS':
                # === FPS down ===
                idx = pointops.farthest_point_sampling(p.contiguous(), o, n_o)  # (m)
                n_p = p[idx.long(), :]  # (m, 3)
                n_f = f[idx.long(), :]  # (m, 12)
            elif self.down_sampling == 'S-FPS':
                # === Dist down ===  vis(n_f[:n_o[0]], 'n_f') vis(n_f[n_o[-2]:n_o[-1]], 'n_f-1')
                idx, n_o = saved_fps(p.contiguous(), f.contiguous(), o, n_o, down_mask)  # (m)
                n_p = p[idx.long(), :]  # (m, 3)
                n_f = f[idx.long(), :]  # (m, 12)
            # =================

            x, _ = pointops.knn_query_and_group(x.contiguous(), p.contiguous(), offset=o, new_xyz=n_p, new_offset=n_o,
                                                nsample=self.nsample, with_xyz=True)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o, f = n_p, n_o, n_f
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
            idx = None
        return [p, x, o, idx, f]


class PointTransformerCls(nn.Module):
    def __init__(self, block, blocks, stride, stages_channel, num_samples, in_channels=10, ds='FPS'):
        super().__init__()
        self.stages_channel = stages_channel
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, stages_channel
        self.down_sampling = ds
        share_planes = 8

        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes,
                                   stride=stride[0], nsample=num_samples[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes,
                                   stride=stride[1], nsample=num_samples[1])  # N/4
        if len(stages_channel) == 3:
            self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes,
                                       stride=stride[2], nsample=num_samples[2])  # N/16

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [TransitionDown(self.in_planes, planes, stride, nsample, self.down_sampling)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, p0, x0, o0, f0, objects_id):
        o0 = o0.int()
        # x0 = p0 if self.in_channels == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1, d1, f1 = self.enc1([p0, x0, o0, f0, objects_id])
        p2, x2, o2, d2, f2 = self.enc2([p1, x1, o1, f1, objects_id])
        if len(self.stages_channel) == 3:
            p3, x3, o3, d3, f3 = self.enc3([p2, x2, o2, f2, objects_id])
            return [p1, p2, p3], [x1, x2, x3], [o1, o2, o3], [d1, d2, d3], [f1, f2, f3]
        else:
            return [p1, p2], [x1, x2], [o1, o2], [d1, d2], [f1, f2]


class PYRAMID(nn.Module):
    def __init__(self, obj_cat_num=160, rel_cat_num=26):
        super().__init__()
        """
        obj-rel: (2 FPS + 2 FPS) / (2 FPS + 4 Dist)
        """
        self.Extractor1 = PointTransformerCls(block=Bottleneck, blocks=[2, 2, 2], stride=[1, 2, 2],
                                              stages_channel=[32, 64, 128], num_samples=[8, 12, 12],
                                              in_channels=9, ds='S-FPS')
        # (x, y, z, rgb, norm)
        self.Extractor2 = PointTransformerCls(block=Bottleneck, blocks=[2, 2], stride=[2, 2],
                                              stages_channel=[256, 512], num_samples=[12, 12],
                                              in_channels=128, ds='S-FPS')
        # ==== Up Sampling Part ====
        self.up_sampling = Up_sampling(input_dims=[256, 128, 64, 32])
        # ==========================

        # ==== Dynamic Pyramid ====
        cls_level_dim = 256
        self.cluster_rule = {32: 15, 64: 7, 128: 3, 256: 2, 512: 1}

        self.dynamic_obj = Pyramid_Transformer(in_planes=512, out_planes=256)
        self.dynamic_rel = Matrix_Transformer(dim=512, heads=8, dim_head=64, dropout=0.)
        # =========================

        # bbox embeddings
        self.obj_bbox_emb = nn.Sequential(nn.Linear(9, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                                          nn.Linear(64, cls_level_dim))

        self.relation_bbox_emb = nn.Sequential(nn.Linear(9, 64), nn.ReLU(inplace=True),
                                               nn.Linear(64, cls_level_dim))

        # cls
        self.obj_cls = nn.Sequential(nn.Linear(cls_level_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                     nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                     nn.Linear(256, obj_cat_num))
        self.rel_cls = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                     nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                     nn.Linear(256, rel_cat_num), nn.Sigmoid())

        self.define_masks()
    def define_masks(self):
        """"""
        batch = 8
        max_points = 10000
        max_ins_per_scene = 9
        max_feature = 512
        self.id_mask = torch.ones((batch * max_ins_per_scene, batch * max_points)).cuda()   # Batch * num_ins_max * max_points
        self.off_mask = self.id_mask * 0     #   num_ins_max * max_points
        self.empty_feat = torch.zeros((batch * max_points, max_feature)).cuda()
        self.zero = torch.tensor([[0]]).cuda()
        self.id_range = torch.range(1, batch * max_ins_per_scene).T.cuda()
        self.p_diff = torch.tensor((10, 10, 10)).cuda()
        self.cluster_mask = torch.zeros((batch * max_ins_per_scene, 100)).cuda()
        self.cluster_feature_mask = torch.zeros((batch * max_ins_per_scene, max_feature)).bool().cuda()
        self.zero_rel_pred = torch.zeros((batch * 72, 26)).float().cuda()

    def forward(self, data_input):
        scene_with_mask, objects_bbox, objects_id, edges, objects_cat, predicate_cat, off_p = data_input
        scene_pc = scene_with_mask[:, :-1]

        p04, x_list_obj, o_list_obj, d_list_obj, f_list_obj = self.Extractor1(scene_pc[:, :3].contiguous(),
                                                                              scene_pc, off_p, scene_with_mask,
                                                                              objects_id)
        p46, x_list_rel, o_list_rel, d_list_rel, f_list_rel = self.Extractor2(p04[-1], x_list_obj[-1], o_list_obj[-1],
                                                                              f_list_obj[-1], objects_id)
        d_list_full = d_list_obj + d_list_rel
        d_list_full[0] = torch.range(0, scene_with_mask.shape[0] - 1).cuda()
        p_list_full = p04 + p46
        x_list_full = x_list_obj + x_list_rel
        o_list_full = o_list_obj + o_list_rel
        f_list_full = f_list_obj + f_list_rel

        """ Up Sampling Part """
        x_list_full = self.up_sampling([p_list_full, x_list_full, o_list_full])
        """"""

        obj_pred, x_feature, pyramid_x, bbox_emb, bbox = self.eff_idx_cls_objs(x_list_full,
                                                                                         scene_with_mask[:, -1],
                                                                                         d_list_full, objects_id,
                                                                                         objects_bbox,
                                                                                         o_list_full, f_list_full)

        rel_pred, y_feature = self.eff_matrix_cls_rels(pyramid_x, edges, bbox_emb, bbox)

        return obj_pred, rel_pred, x_feature, y_feature

    def eff_idx_cls_objs(self, x_list_full, mask, d_list_full, objects_id, bbox, o_list_full, f=None):
        """
        x_list_full: point features in every layers
        mask: point mask to identify instances
        d_list_full: sub-sample index in every downs-sampling layers
        objects_id: instances' id in every batches
        bbox: instances' bbox message
        o_list_full: offsets in every layers
        f: org point features, using in testing
        """
        pyramid_x = []  # collecting features under every scales

        id_flat = torch.cat(objects_id)

        mask_full = []
        for d in d_list_full:  # find the obj_mask in this scale
            mask = mask[d.long()]
            mask_full.append(mask)
        for scale in range(len(x_list_full)):   # circle in each scales
            x_scale = x_list_full[scale]
            f_scale = f[scale]
            mask_scale = mask_full[scale]
            off_scale = o_list_full[scale]
            # ====== find instances points =======
            id_mask = self.id_mask[:id_flat.shape[0], :x_scale.shape[0]].clone()
            id_mask = id_mask * id_flat[:, None]

            off_mask_full = self.off_mask[:id_flat.shape[0], :x_scale.shape[0]].clone()
            row = 0
            for i in range(len(off_scale)):
                if i == 0:
                    s_i, e_i = 0, off_scale[0]
                else:
                    s_i, e_i = off_scale[i - 1], off_scale[i]
                off_mask_full[row:row + objects_id[i].shape[0], s_i:e_i] = 1
                row = row + objects_id[i].shape[0]

            assert torch.sum(mask_scale == 0) == 0   # avoid bug
            id_mask_off = id_mask * off_mask_full
            instance_bool = mask_scale[None, :] == id_mask_off
            assert torch.sum(instance_bool, dim=0).max() == 1
            # ===== get instances ======
            knn_num = torch.tensor([self.cluster_rule[x_scale.shape[1]]]).to(id_flat.device)

            # batch * max_points, 512 / batch * max_points, 10 / no need / no need / batch * max_ins, 512
            instance_range = self.id_range[:id_flat.shape[0]]
            range_bool = instance_range[:, None] * instance_bool  # ins_num, points
            range_bool = torch.sum(range_bool, dim=0)
            zero_bool = torch.sum(range_bool == 0)  # other points number
            ins_id, ins_idx = torch.sort(range_bool)
            sort_idx = ins_idx[zero_bool:]

            instance_features = x_scale[sort_idx]
            instance_points = f_scale[sort_idx]

            instance_numbers = torch.sum(instance_bool, dim=1)

            instances_off = torch.cumsum(instance_numbers, dim=0)
            instances_n_off = self.id_range[:id_flat.shape[0]] * knn_num

            b_extended = torch.cat([self.zero[0], instances_off])
            splits = [instance_features[start:end] for start, end in zip(b_extended[:-1], b_extended[1:])]
            x_means = torch.stack([torch.sum(split, dim=0) / split.shape[0] for split in splits])

            x_in_scale = self.eff_cluster_pooling(instance_features, instance_points, instances_off, instances_n_off,
                                                  x_means, scale, ins_id[zero_bool:])

            pyramid_x.append(x_in_scale)  # B, C
        objects_bbox_embedding = self.obj_bbox_emb(bbox)  # obj * dim * 1

        '''pyramid feature merge'''
        pyramid_stage = torch.stack(pyramid_x[:4]).permute(1, 0, 2)  # obj, pyramid, dim
        x_feature = self.dynamic_obj(pyramid_stage, objects_bbox_embedding, bbox)

        x_pred = self.obj_cls(x_feature)
        return x_pred, x_feature, pyramid_x, objects_bbox_embedding, bbox

    def eff_matrix_cls_rels_filter(self, pyramid_x, edges, bbox_emb, bbox):
        """
        Efficient relation classification with candidate edge filtering.

        Args:
            pyramid_x: [N, 5, 512] instance features
            edges: [E, 2] long tensor of subject-object index pairs
            bbox_emb: [N, C] bbox-level embedding
            bbox: [N, 9] bbox geometry

        Returns:
            rel_pred: [E, rel_cls] relation logits for all edges
            rel_matrix: [E', dim] sparse relation features (for valid edges only)
        """
        pyramid_x = torch.stack(pyramid_x, dim=1)  # [N, 5, 512]

        E_total = edges.shape[0]
        # Extracting indices for edges
        subj_indices = edges[:, 0]
        obj_indices = edges[:, 1]

        # Extracting instance features
        subject_ins = pyramid_x[subj_indices, 1:, :]  # edges, 4, 512
        object_ins = pyramid_x[obj_indices, 1:, :]  # edges, 4, 512

        # === Pair Filtering === #
        keep_mask = filter_candidate_pairs(subject_ins.reshape((E_total, -1)), object_ins.reshape((E_total, -1)), tau_rate=0.7, min_edges=50)  # shape [E], bool or byte
        keep_edge_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

        subj_keep = subj_indices[keep_edge_idx]
        obj_keep = obj_indices[keep_edge_idx]

        subj_ins = pyramid_x[subj_keep, 1:, :]  # [E', 4, 512]
        obj_ins = pyramid_x[obj_keep, 1:, :]  # [E', 4, 512]

        subj_emb = bbox_emb[subj_keep]
        obj_emb = bbox_emb[obj_keep]

        delta_bbox = bbox[subj_keep] - bbox[obj_keep]
        rel_bbox_embeddings = self.relation_bbox_emb(delta_bbox)

        subj_dot_obj = subj_ins.unsqueeze(1) + obj_ins.unsqueeze(2)  # [E', 4, 4, 512]
        rel_matrix_sparse = self.dynamic_rel(subj_ins, obj_ins, subj_dot_obj.view(subj_keep.size(0), -1, 512),
                                             subj_emb, obj_emb, rel_bbox_embeddings)

        rel_logits_sparse = self.rel_cls(rel_matrix_sparse)  # [E', rel_cls]

        rel_pred = self.zero_rel_pred[:E_total, :].clone()
        rel_pred[keep_edge_idx] = rel_logits_sparse

        return rel_pred, rel_matrix_sparse

    def eff_cluster_pooling(self, instances_features, instances_points, instances_offs, instances_n_offs, x_mean, scale, ins_diff):
        """ FPS + Cluster + Pooling + Cat """
        ins_number, ins_dim = instances_offs.shape[0], instances_features.shape[-1]
        if ins_dim < 256:
            p = instances_points[:, :3]
            p_ins_diff = self.p_diff * ins_diff[:, None]

            p = p + p_ins_diff
            instances_offs_diff = instances_offs - torch.cat([self.zero[0], instances_offs[:-1]])

            # === FPS down ===
            idx = pointops.farthest_point_sampling(p.contiguous(), instances_offs, instances_n_offs.long())  # (m)
            n_p = p[idx.long(), :].contiguous()  # (m, 3)

            x_k, idx = pointops.knn_query_and_group(instances_features.contiguous(), p.contiguous(), instances_offs,
                                                    new_xyz=n_p, new_offset=instances_n_offs,
                                                    nsample=min(int(instances_offs_diff.median()), 100),
                                                    with_xyz=False)   # ins*knn, samp, feat

            x_k = x_k.view(ins_number, -1, min(int(instances_offs_diff.median()), 100), ins_dim)
            # ins, key, samp, feat  (70, 15, 45, 32)
            ins_mask = self.cluster_mask[:ins_number, :min(int(instances_offs_diff.median()), 100)].clone()
            ins_use_full_mean_mask = self.cluster_feature_mask[:ins_number, :].clone()  # for ins_points < knn
            for ins in range(instances_offs_diff.shape[0]):
                if instances_offs_diff[ins] < self.cluster_rule[ins_dim]:
                    ins_use_full_mean_mask[ins, :] = True
                # ins_mask[ins, :instances_offs_diff[ins]] = 1
                ins_mask[ins, :max(int(instances_offs_diff[ins] / self.cluster_rule[ins_dim]), 1)] = 1
            x_k = x_k.permute(0, 2, 1, 3) * ins_mask[:, :, None, None]

            x_k = torch.sum(x_k, dim=1) / torch.sum(ins_mask, dim=1)[:, None, None]   # (70, 15, 32)
            x_k = x_k.view(ins_number, -1)

            x_k = torch.cat([x_k, x_mean], dim=-1)   # add a full pooling first
            x_k_full_mean = x_mean.tile(1, int(512 / x_mean.shape[1]))
            x_cluster = x_k * ~ins_use_full_mean_mask + x_k_full_mean * ins_use_full_mean_mask
        else:
            x_cluster = x_mean.tile(1, int(512 / x_mean.shape[1]))
        return x_cluster

