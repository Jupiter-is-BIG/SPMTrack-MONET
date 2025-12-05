import torch
import math
import torch.nn as nn
from timm.layers import trunc_normal_

from .merge import _tmoe_merge, _tmoe_unmerge


class TMoELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int, alpha: float, dropout: float,
                 rs_expert: bool = False, init_method: str = 'bert',
                 expert_nums: int = 4, blc_alpha: float = 0.0, blc_weight: float = 0.0,
                 shared_expert: bool = False, route_compression: bool = False,
                 top_k: int | None = None):
        super().__init__()
        self.top_k = top_k
        self.route_compression = route_compression
        if self.route_compression:
            self.compress_expert = nn.ParameterList([nn.Parameter(torch.empty(r, in_dim)) for _ in range(expert_nums)])
        else:
            self.compress_expert = nn.Parameter(torch.empty(r, in_dim))
        self.enable_shared_expert = shared_expert
        assert not (self.route_compression and self.enable_shared_expert)
        if shared_expert:
            self.shared_expert = nn.Parameter(torch.empty(out_dim, r))
        self.routed_experts = nn.ParameterList([nn.Parameter(torch.empty(out_dim, r)) for _ in range(expert_nums)])
        self.r = r
        self.alpha = alpha
        self.rs_expert = rs_expert
        self.expert_num = expert_nums
        self.expert_route = nn.Linear(in_dim, self.expert_num, bias=False)

        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight

        if rs_expert:
            self.scaling = alpha / math.sqrt(r)
        else:
            self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if init_method == 'lora':
            if self.route_compression:
                for compress_expert in self.compress_expert:
                    nn.init.kaiming_uniform_(compress_expert, a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.compress_expert, a=math.sqrt(5))
            if self.enable_shared_expert:
                nn.init.zeros_(self.shared_expert)
            for routed_expert in self.routed_experts:
                nn.init.zeros_(routed_expert)
        elif init_method == 'gaussian':
            if self.route_compression:
                for compress_expert in self.compress_expert:
                    nn.init.normal_(compress_expert, std=1. / self.r)
            else:
                nn.init.normal_(self.compress_expert, std=1. / self.r)
            if self.enable_shared_expert:
                nn.init.zeros_(self.shared_expert)
            for routed_expert in self.routed_experts:
                nn.init.zeros_(routed_expert)
        elif init_method == 'bert':
            if self.route_compression:
                for compress_expert in self.compress_expert:
                    trunc_normal_(compress_expert, std=.02)
            else:
                trunc_normal_(self.compress_expert, std=.02)
            if self.enable_shared_expert:
                trunc_normal_(self.shared_expert, std=.02)
            for routed_expert in self.routed_experts:
                trunc_normal_(routed_expert, std=.02)
        else:
            raise ValueError(f'Unknown init method: {init_method}')

        #nn.init.kaiming_uniform_(self.expert_route.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        route_weight = self.expert_route(x).softmax(dim=-1)
        if self.route_compression:
            results = []
            for i in range(self.expert_num):
                compress_expert_out = self.dropout(x) @ self.compress_expert[i].transpose(0, 1)
                results.append(torch.unsqueeze(route_weight[:,:,i], -1) * (compress_expert_out @ self.routed_experts[i].transpose(0, 1)) * self.scaling)
        else:
            results = []
            compress_expert_out = self.dropout(x) @ self.compress_expert.transpose(0, 1)
            if self.enable_shared_expert:
                results.append((compress_expert_out @ self.shared_expert.transpose(0, 1)) * self.scaling)
            for i in range(self.expert_num):
                results.append(torch.unsqueeze(route_weight[:,:,i], -1) * (compress_expert_out @ self.routed_experts[i].transpose(0, 1)) * self.scaling)
        return sum(results)
    
    def forward(self, x: torch.Tensor):
        logits = self.expert_route(x)  # [B, T, E]

        if self.top_k is None or self.top_k >= self.expert_num:
            route_weight = logits.softmax(dim=-1)
            use_topk = False
        else:
            topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)  # [B, T, K]
            topk_weight = topk_vals.softmax(dim=-1)                         # [B, T, K]
            use_topk = True

        if self.route_compression:
            route_weight = logits.softmax(dim=-1) if not use_topk else _build_full_route(logits, topk_idx, topk_weight)
            results = []
            for i in range(self.expert_num):
                compress_expert_out = self.dropout(x) @ self.compress_expert[i].transpose(0, 1)
                expert_out = compress_expert_out @ self.routed_experts[i].transpose(0, 1)
                gate = route_weight[:, :, i].unsqueeze(-1)
                results.append(gate * expert_out * self.scaling)
            return sum(results)

        # ---- non-route_compression optimized top-k ----
        compress_expert_out = self.dropout(x) @ self.compress_expert.transpose(0, 1)  # [B, T, r]

        results = []

        if self.enable_shared_expert:
            shared_out = compress_expert_out @ self.shared_expert.transpose(0, 1)
            results.append(shared_out * self.scaling)

        if not use_topk:
            route_weight = logits.softmax(dim=-1)
            for i in range(self.expert_num):
                expert_out = compress_expert_out @ self.routed_experts[i].transpose(0, 1)
                gate = route_weight[:, :, i].unsqueeze(-1)
                results.append(gate * expert_out * self.scaling)
            return sum(results)

        # --- top-k optimized path ---
        B, T, K = topk_idx.shape
        flat_idx = topk_idx.view(B * T, K)        # [BT, K]
        flat_weight = topk_weight.view(B * T, K)  # [BT, K]
        flat_ce = compress_expert_out.view(B * T, -1)  # [BT, r]

        out = torch.zeros(B * T, self.routed_experts[0].shape[0], device=x.device, dtype=x.dtype)  # [BT, out_dim]

        for k_i in range(K):
            expert_ids = flat_idx[:, k_i]  # [BT]
            w_k = flat_weight[:, k_i].unsqueeze(-1)  # [BT, 1]
            expert_out_k = torch.zeros_like(out)
            for e_id in range(self.expert_num):
                mask = (expert_ids == e_id)
                if not mask.any():
                    continue
                ce_sel = flat_ce[mask]  # [N_e, r]
                routed = self.routed_experts[e_id]  # [out_dim, r]
                expert_out_k[mask] = ce_sel @ routed.transpose(0, 1)  # [N_e, out_dim]

            out += w_k * expert_out_k * self.scaling

        out = out.view(B, T, -1)
        results.append(out)
        return sum(results)


class LinearWithTMoE(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float, rs_expert: bool=False, init_method: str='bert',
                 merge_on_eval: bool = False, expert_nums: int = 4, shared_expert: bool = False, route_compression: bool = False):
        super().__init__()
        self.linear = linear
        self.tmoe = TMoELayer(linear.in_features, linear.out_features, r, alpha, dropout, rs_expert, init_method, expert_nums=expert_nums, shared_expert=shared_expert, route_compression=route_compression)
        self.merge_on_eval = merge_on_eval
        self.merged = False

    def forward(self, x: torch.Tensor):
        return self.linear(x) + self.tmoe(x)

class LinearWithTMoE_TimmQKV(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float, rs_expert: bool = False,
                 init_method: str = 'bert',
                 target_q: bool = True, target_k: bool = True, target_v: bool = True,
                 expert_nums: int = 4, shared_expert: bool = False, route_compression: bool = False):
        super().__init__()
        dim = linear.in_features
        bias = linear.bias is not None
        q = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        k = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        v = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        q.weight.data.copy_(linear.weight.data[:dim])
        k.weight.data.copy_(linear.weight.data[dim:2*dim])
        v.weight.data.copy_(linear.weight.data[2*dim:])
        q.weight.requires_grad = k.weight.requires_grad = v.weight.requires_grad = linear.weight.requires_grad
        if bias:
            q.bias.data.copy_(linear.bias.data[:dim])
            k.bias.data.copy_(linear.bias.data[dim:2*dim])
            v.bias.data.copy_(linear.bias.data[2*dim:])
            q.bias.requires_grad = k.bias.requires_grad = v.bias.requires_grad = linear.bias.requires_grad

        if target_q:
            self.q = LinearWithTMoE(q, r, alpha, dropout, rs_expert, init_method, expert_nums=expert_nums, shared_expert=shared_expert, route_compression=route_compression)
        else:
            self.q = q
        if target_k:
            self.k = LinearWithTMoE(k, r, alpha, dropout, rs_expert, init_method, expert_nums=expert_nums, shared_expert=shared_expert, route_compression=route_compression)
        else:
            self.k = k
        if target_v:
            self.v = LinearWithTMoE(v, r, alpha, dropout, rs_expert, init_method, expert_nums=expert_nums, shared_expert=shared_expert, route_compression=route_compression)
        else:
            self.v = v

    def forward(self, x: torch.Tensor):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return torch.cat((q, k, v), dim=-1)
