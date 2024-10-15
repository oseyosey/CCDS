from typing import Optional

import math

import torch

from ccds.training.data_selection.embed.dist import gather_sum, get_rank, get_world_size

def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, _S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def slice_sparse_tensor_rows(t: torch.sparse.Tensor, min_row: int, max_row: int) -> torch.sparse.Tensor:
    assert min_row < max_row, f"can't slice from row {min_row} to {max_row}"
    t = t.coalesce()
    row_idxs = t.indices()[0]
    index_mask = (min_row <= row_idxs) & (row_idxs < max_row)

    num_rows = (max_row - min_row)
    num_cols = t.shape[1]

    idxs = t.indices()[:, index_mask]
    vals = t.values()[index_mask]
    return torch.sparse_coo_tensor(idxs, vals, size=(num_rows, num_cols)).coalesce()


def slice_tensor_rows(t: torch.Tensor, min_row: int, max_row: int) -> torch.Tensor:
    if t.is_sparse:
        return slice_sparse_tensor_rows(t=t, min_row=min_row, max_row=max_row)
    else:
        return t[min_row:max_row]


@torch.no_grad
def maxsim(
    X: torch.Tensor, y: torch.Tensor, 
    maximize: bool, chunk_size: int = 8_000,
    debug_mem_usage: bool = False) -> torch.Tensor:
    device = X.device
    n_samples = X.shape[0]

    max_sim_v = torch.zeros(n_samples, device=device, dtype=X.dtype)
    max_sim_i = torch.zeros(n_samples, device=device, dtype=torch.int64)

    # TODO: Implement faster max (without going to dense tensors).
    # TODO: Use multiple GPUs.
    rank = get_rank()
    world_size = get_world_size()

    worker_worklist_size = int(math.ceil(n_samples / world_size))
    splits_start_idx = worker_worklist_size * rank
    splits_end_idx = worker_worklist_size * (rank + 1)

    for i in range(splits_start_idx, splits_end_idx, chunk_size):
        start, end = i, min(i + chunk_size, n_samples)
        sub_x = slice_tensor_rows(X, start, end)
        if debug_mem_usage: print(f"[maxsim] step {i} cuda mem free/total = {torch.cuda.mem_get_info()}")
        if debug_mem_usage: print("[maxsim] sub_x.shape:", sub_x.shape, "//", "y.shape:", y.shape)
        sub_sim = sub_x @ y # TODO – Implement sparse max here to save mem!
        sub_sim = sub_sim
        if maximize:
            sub_max_sim_v, sub_max_sim_i = sub_sim.to_dense().max(dim=-1)
        else:
            sub_max_sim_v, sub_max_sim_i = sub_sim.to_dense().min(dim=-1)
        del sub_sim
        del sub_x
        torch.cuda.empty_cache() # needs to happen after maxsim for some reason.
        max_sim_v[start: end] = sub_max_sim_v
        max_sim_i[start: end] = sub_max_sim_i
    
    # gather
    max_sim_v = gather_sum(max_sim_v)
    max_sim_i = gather_sum(max_sim_i)
    k = y.shape[1]

    assert max_sim_v.shape == (n_samples,)
    assert max_sim_i.shape == (n_samples,)
    assert max_sim_i.min() >= 0
    assert max_sim_i.max() <= k

    return max_sim_v, max_sim_i


def forward_batched(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        dataset_input_ids: Optional[torch.Tensor] = None,
        dataset_attention_mask: Optional[torch.Tensor] = None,
        **second_stage_model_kwargs,
) -> torch.Tensor:
    if hasattr(model, "module"):
        model = model.module
    
    if hasattr(model, "first_stage_model"):
        # Support pooling over 3D dataset_input_ids inputs.
        if len(dataset_input_ids.shape) == 2:
            dataset_input_ids = dataset_input_ids[None]
            dataset_attention_mask = dataset_attention_mask[None]

        dataset_embeddings = []
        for j in range(len(dataset_input_ids)):
            i = 0
            dataset_embeddings_batch = []
            while i < dataset_input_ids.shape[1]:
                dataset_embeddings_batch.append(
                    model.first_stage_model(
                        input_ids=dataset_input_ids[j][i:i+batch_size],
                        attention_mask=dataset_attention_mask[j][i:i+batch_size],
                    )
                )
                i += batch_size
            dataset_embeddings.append(
                torch.cat(dataset_embeddings_batch, dim=0)
            )
       
        # Automatically pool over 3D dataset_input_ids.
        dataset_embeddings = torch.stack(dataset_embeddings, dim=0).mean(dim=0)

        j = 0
        outputs = []
        while j < len(input_ids):
            outputs.append(
                model.second_stage_model(
                    input_ids=input_ids[j:j+batch_size],
                    attention_mask=attention_mask[j:j+batch_size],
                    dataset_embeddings=dataset_embeddings,
                    **second_stage_model_kwargs,
                )
            )
            j += batch_size
        return torch.cat(outputs, dim=0)

    else:
        i = 0
        outputs = []
        while i < len(input_ids):
            outputs.append(
                model(
                    input_ids=input_ids[i:i+batch_size],
                    attention_mask=attention_mask[i:i+batch_size],
                    **second_stage_model_kwargs,
                )
            )
            i += batch_size
        return torch.cat(outputs, dim=0)