from typing import Union
import dgl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from utils import NodeType
from metrics import graph_cosine_similiarity

from .dataset import TreeHopInferenceDataset


class ResNet(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, input_size)
        self.activate = nn.ReLU()
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        # post Norm
        x_norm = self.layer_norm(x)
        x = x + self.activate(self.linear(x_norm))
        return x


class MultiMLPLayer(nn.Module):
    def __init__(
        self,
        input_size,
        mlp_size,
        num_layers: int = 1
    ):
        super(MultiMLPLayer, self).__init__()
        self.layers = nn.Sequential()

        for i in range(num_layers):
            if i == 0 and input_size != mlp_size:
                self.layers.append(nn.Linear(input_size, mlp_size))
                self.layers.append(ResNet(mlp_size))
            else:
                self.layers.append(ResNet(mlp_size))

    def forward(self, x):
        x_out = self.layers(x)
        return x_out


class AttentionHead2D(nn.Module):
    def __init__(
        self,
        input_size,
        attn_size,
        mlp_size,
        *,
        bias=True,
        num_mlp=1,
        dropout=0.1
    ):
        super(AttentionHead2D, self).__init__()
        self.W_Q = nn.Linear(input_size, attn_size, bias=bias)
        self.W_K = nn.Linear(input_size, attn_size, bias=bias)
        self.W_V = nn.Linear(input_size, attn_size, bias=bias)

        # self.activate = nn.ReLU()
        self.mlp = MultiMLPLayer(attn_size, mlp_size, num_layers=num_mlp)
        self.dropout = nn.Dropout(dropout)
        self.mlp_scale = nn.Linear(mlp_size, attn_size, bias=bias)

    def forward(self, Q, K, V):
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        if Q.dim() == 3:
            QK = torch.einsum("bud,bud->bd", Q, K)
        elif Q.dim() == 2:
            QK = Q * K
        else:
            raise IndexError(f"Not a supported input dimension: {Q.dim()}")

        # instead of matmul in 3D, use elementwise mul, then normalize
        scores = QK / Q.shape[1] ** 0.5
        attn = F.softmax(scores, dim=-1)
        attn_out = self.dropout(attn) * V

        mlp_out = self.mlp(attn_out)

        return self.mlp_scale(mlp_out) + attn_out


class MultiHeadAttention2D(nn.Module):
    def __init__(
        self,
        input_size,
        attn_size,
        mlp_size,
        num_mlp=1,
        num_heads=1,
        bias=True,
        dropout=0.1
    ):
        if not isinstance(num_heads, int) or num_heads < 1:
            raise ValueError("num_heads must be a positive integer")

        super(MultiHeadAttention2D, self).__init__()

        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            AttentionHead2D(input_size, attn_size, mlp_size,
                            num_mlp=num_mlp, bias=bias, dropout=dropout)
            for _ in range(num_heads)
        ])

    def forward(self, Q, K, V):
        lst_attn_out = []
        for attn_head in self.heads:
            out = attn_head(Q, K, V)
            lst_attn_out.append(out)

        attn_out = torch.cat(lst_attn_out, dim=-1)
        return attn_out


class TreeHopNode(nn.Module):
    def __init__(self, embed_size, g_size, mlp_size, n_mlp=1, n_head=1):
        super(TreeHopNode, self).__init__()
        self.update_gate = MultiHeadAttention2D(
            embed_size, g_size, mlp_size,
            num_heads=n_head,
            num_mlp=n_mlp,
            dropout=0.
        )
        self.update_attn_scale = nn.Linear(g_size * n_head, embed_size, bias=False)

    def reduce_func(self, nodes):
        # message passing
        Q = nodes.mailbox["q"].clone().squeeze(1)         # last query
        K = nodes.data["rep"]           # this ctx
        V_update = nodes.data["rep"]           # this ctx

        update_gate = self.update_gate(Q, K, V_update)

        h = Q - K + self.update_attn_scale(update_gate)
        return {"h": h}


class TreeHopModel(
        nn.Module,
        PyTorchModelHubMixin,
        library_name="treehop-rag",
        license="mit",
        tags=["Retrieval-Augmented Generation", "Information Retrieval", "multi-hop question answering"],
        repo_url="https://github.com/allen-li1231/treehop-rag/",
        paper_url="https://arxiv.org/abs/2504.20114",
    ):
    def __init__(
        self,
        x_size,
        g_size,
        mlp_size,
        n_mlp=3,
        dropout=0.1,
        n_head=3
    ):
        super(TreeHopModel, self).__init__()
        self.n_head = n_head

        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.node = TreeHopNode(
            x_size, g_size, mlp_size, n_mlp=n_mlp, n_head=n_head
        )

        self._graphs = None
        self.device = torch.device("cpu")

    def forward(self, g: dgl.DGLGraph):
        """Compute graph-reranker prediction given a graph.

        Parameters
        ----------
        batch : dgl.graph

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # propagate
        if "y" in g.ndata:
            mask_query = g.ndata["y"] != NodeType.query.value
            g.ndata["h"] = F.normalize(g.ndata["h"], dim=-1)
            dgl.prop_nodes_topo(
                g,
                reverse=False,
                message_func=dgl.function.copy_u('h', 'q'),
                reduce_func=self.node.reduce_func,
                # apply_node_func=self.node.apply_node_func,
            )

            h = g.ndata["h"][mask_query]
        elif "mask" in g.edata:
            idx_prop_edges = g.edata["mask"].nonzero().flatten()
            if len(idx_prop_edges) == 0:
                return

            _, mask_query = g.find_edges(idx_prop_edges)
            g.prop_edges(
                idx_prop_edges,
                message_func=dgl.function.copy_u('h', 'q'),
                reduce_func=self.node.reduce_func,
            )
            h = g.ndata["h"][mask_query]
        else:
            raise LookupError("Unable to determine training or inference target")

        h = self.dropout(h)
        return h

    def to(self, device):
        super(TreeHopModel, self).to(device)
        if self._graphs is not None:
            self._graphs = [graph.to(device) for graph in self._graphs]

        self.device = torch.device(device)
        return self

    def calc_sim_prob(self, graph: dgl.DGLGraph):
        mask_last_layer_edges = graph.edata["mask"]
        sim = graph_cosine_similiarity(graph)
        sim = sim[mask_last_layer_edges]

        p = F.softmax(sim, dim=-1)
        graph.edata["p"][mask_last_layer_edges] = p
        return p

    def _get_filtered_query_embs(self, query_masks, top_n, inplace=False):
        # can be multi-processed

        idx_queries, query_embs = [], []
        for mask, g in zip(query_masks, self._graphs):
            if not inplace:
                mask_query_edges = g.edata["mask"].detach().clone()
            else:
                mask_query_edges = g.edata["mask"]

            if isinstance(mask, list):
                mask = np.asarray(mask)
        
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            mask = mask.to(self.device)
            query_mask = mask.any(axis=-1)
            if not query_mask.any():
                idx_queries.append(None)
                continue

            idx_query_edges = g.edges("eid")[mask_query_edges.nonzero().flatten()]
            if mask_query_edges.all():
                # for silenting torch error
                mask_query_edges.copy_(mask.ravel())
            else:
                # size of query edges should match with query_status
                mask_query_edges.masked_scatter_(mask_query_edges, mask.ravel())

            # get last query edge ids
            idx_query_edges = g.edges("eid")[mask_query_edges.nonzero().flatten()]
            _, query_nodes = g.find_edges(idx_query_edges)
            # resize to match number of ctx embs (retrieved passages)
            query_nodes = query_nodes.repeat(top_n, 1).T.flatten()
            idx_queries.append(query_nodes)
            query_embs.append(g.ndata["h"][query_nodes])

        query_embs = torch.cat(query_embs)
        return idx_queries, query_embs

    @torch.inference_mode
    def next_query(
        self,
        *,
        ctx_embs: Union[torch.Tensor, np.ndarray],
        q_emb: Union[torch.Tensor, np.ndarray, None] = None,
        query_passage_masks: Union[torch.Tensor, np.ndarray, None] = None,
        batch_size=1024,
        num_workers=0
    ):
        if q_emb is None and self._graphs is None:
            raise ValueError("query embedding must set to initialize graph.")

        if q_emb is not None and self._graphs is not None:
            raise ValueError("query embedding presents when graph exists.")

        if q_emb is not None and query_passage_masks is not None:
            raise ValueError("query embedding is provided along with query_passage_masks.")

        if q_emb is None and query_passage_masks is None:
            raise ValueError("either query embedding or query_passage_masks should be provided.")

        if q_emb is None and query_passage_masks is not None:
            assert len(query_passage_masks) == len(self._graphs)
            top_n = query_passage_masks[0].shape[1]
            idx_last_queries, q_emb = \
                self._get_filtered_query_embs(query_passage_masks, top_n=top_n, inplace=True)

        assert q_emb.shape[-1] == ctx_embs.shape[-1], \
            "query and context embedding dimension must match"

        if isinstance(q_emb, np.ndarray):
            q_emb = torch.from_numpy(q_emb)
        if isinstance(ctx_embs, np.ndarray):
            ctx_embs = torch.from_numpy(ctx_embs)

        if q_emb.dim() == 1:
            if ctx_embs.dim() == 1:
                if self._graphs is None:
                    lst_node_out, lst_node_in = [[0]], [[1]]
                    ary_rep = torch.stack([q_emb, ctx_embs]).unsqueeze(0)
                    ary_h = q_emb.repeat(2, 1).unsqueeze(0)
                else:
                    lst_node_out = idx_last_queries
                    lst_node_in = [[g.num_nodes()] for g in self._graphs]
                    ary_h = q_emb.repeat(1, 1, 1)

            elif ctx_embs.dim() == 2:
                if self._graphs is None:
                    lst_node_out = [[0]] * len(ctx_embs)
                    lst_node_in = [[1]] * len(ctx_embs)
                    ary_rep = [torch.stack([q_emb.repeat(1, 1), ctx]) for ctx in ctx_embs]
                    ary_h = q_emb.repeat(len(ctx_embs), 2, 1)
                else:
                    raise IndexError(f"Number of context embeddings mismatches with "
                                     f"previous number of query ({ctx_embs.shape[0]} vs 1)")
            else:
                raise IndexError(f"Too many dimensions for context embeddings "
                                 f"{ctx_embs.shape} to match query embeddings {q_emb.shape}")

        elif q_emb.dim() == 2:
            if q_emb.shape[0] != ctx_embs.shape[0]:
                raise IndexError(f"Number of query and context embedding must match "
                                 f"({q_emb.shape[0]} vs {ctx_embs.shape[0]})")
            if self._graphs is None:
                n_nodes = 1 + ctx_embs.shape[1]
                lst_node_out = [[0] * ctx_embs.shape[1]] * len(ctx_embs)
                lst_node_in = [list(range(1, n_nodes)) for _ in range(len(ctx_embs))]
                ary_rep = [torch.cat([q.unsqueeze(0), ctx]) for q, ctx in zip(q_emb, ctx_embs)]
                ary_h = q_emb.repeat(1, n_nodes).view(len(ctx_embs), n_nodes, -1)
            else:
                lst_node_out = idx_last_queries
                lst_node_in = [list(range(g.num_nodes(), g.num_nodes() + len(out)))
                                if out is not None else None
                                for g, out in zip(self._graphs, lst_node_out)]
                ary_rep = ctx_embs
                ary_h = q_emb
        else:
            raise IndexError(f"Expect 1 or 2 dimensions for query embeddings, got {q_emb.shape}")

        if self._graphs is None:
            self._graphs = []
            for node_out, node_in, rep, h in zip(lst_node_out, lst_node_in,
                                                 ary_rep, ary_h):
                if node_out is None:
                    continue

                g = dgl.graph(
                    (node_out, node_in),
                    idtype=torch.int64
                )
                # node data
                g.ndata["rep"], g.ndata["h"] = rep, h
                # edge data
                g.edata["mask"] = torch.ones(g.num_edges(), dtype=torch.bool)

                self._graphs.append(g.to(self.device))
        else:
            i_emb = 0
            for g, node_out, node_in, idx_last_q in zip(self._graphs,
                                                        lst_node_out,
                                                        lst_node_in,
                                                        idx_last_queries):
                if idx_last_q is None or node_out is None:
                    g.edata["mask"].zero_()
                    continue

                ctx_data = {
                    'rep': ary_rep[i_emb: i_emb + len(idx_last_q)].view(-1, ary_rep.shape[-1]),
                    'h': ary_h[i_emb: i_emb + len(idx_last_q)].view(-1, ary_rep.shape[-1])
                }
                edge_data = {
                    'mask': torch.ones(len(node_out), dtype=torch.bool)
                }
                g.add_nodes(len(node_in), data=ctx_data)
                # set last layer to zero and append the next layer to the tree
                g.edata["mask"].zero_()
                g.add_edges(node_out, node_in, data=edge_data)
                i_emb += len(idx_last_q)

        next_q_embs = self.batch_inference(
            self._graphs,
            batch_size=batch_size,
            num_workers=num_workers
        )
        # normalize inplace
        next_q_embs = F.normalize(next_q_embs, dim=-1, out=next_q_embs)

        # p = torch.cat(list(map(self.calc_sim_prob, self._graphs)))
        # if return_sim_prob:
        #     return next_q_embs, p

        return next_q_embs

    @torch.inference_mode
    def batch_inference(
        self,
        graphs,
        batch_size=1024,
        num_workers=0
    ):
        dataset = TreeHopInferenceDataset(graphs)
        data_loader = dgl.dataloading.GraphDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
        )

        self.eval()
        lst_next_q_embs = []
        lst_graphs = []
        for i, batch in enumerate(data_loader):
            next_query = self.forward(batch.to(self.device))
            if next_query is None:
                continue

            lst_next_q_embs.append(next_query)
            lst_graphs.extend(dgl.unbatch(batch.to("cpu")))

        # sync intermediate query embeddings
        self._graphs = lst_graphs
        next_q_embs = torch.cat(lst_next_q_embs)
        return next_q_embs

    def reset_query(self):
        """Reset the graph reranker."""
        self._graphs = None
        if self.device.type != "cpu":
            getattr(torch, self.device.type).empty_cache()
