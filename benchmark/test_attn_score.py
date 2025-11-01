import torch
import triton
import triton.language as tl

@triton.jit
def triton_fill_flashomni_kernel(sparse_kv_info, sparse_info, 
                                 num_kv_to_select, sorted_kv_indices, 
                                 num_to_select, sorted_indices, 
                                 NK: tl.constexpr, T2T: tl.constexpr = 0):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    if q == 0:
        # process_sparse_info
        cur_num_to_select = tl.load(num_to_select + b)
        cur_sorted_idx_ptr = sorted_indices + b * NK
        cur_final_map_ptr = sparse_info + b * H * NK + h * NK + T2T
        for i in range(cur_num_to_select):
            cur_idx = tl.load(cur_sorted_idx_ptr + i)
            tl.store(cur_final_map_ptr + cur_idx, 0)
    # process_sparse_kv_info
    elif q >= T2T:
        cur_num_kv_to_select = tl.load(num_kv_to_select + b * H * Q + h * Q + q)
        cur_sorted_kv_idx_ptr = sorted_kv_indices + b * H * Q * NK + h * Q * NK + q * NK
        cur_sparse_kv_info_ptr = sparse_kv_info + b * H * Q * NK + h * Q* NK + q * NK 
        for i in range(cur_num_kv_to_select):
            cur_idx = tl.load(cur_sorted_kv_idx_ptr + i)
            tl.store(cur_sparse_kv_info_ptr + cur_idx, 0)

def fill_flashomni_triton(sparse_kv_info, num_kv_to_select, sorted_kv_indices,
                          sparse_info, num_q_to_select, sorted_q_indices, T2T=0):
    sparse_kv_info = sparse_kv_info.contiguous()
    num_kv_to_select = num_kv_to_select.contiguous()
    sorted_kv_indices = sorted_kv_indices.contiguous()
    B, H, NBlock_Q, NBlock_KV = sparse_kv_info.shape

    sparse_info = sparse_info.contiguous()
    num_q_to_select = num_q_to_select.contiguous()
    sorted_q_indices = sorted_q_indices.contiguous()
    B_q, H_q, num_kv = sparse_info.shape
    assert B == B_q and H == H_q and num_kv == NBlock_KV and NBlock_Q == NBlock_KV, "Batch and Head dimensions must match for sparse_info and sparse_kv_info"
    grid = (B, H, NBlock_Q)
    triton_fill_flashomni_kernel[grid](sparse_kv_info, sparse_info, 
                                       num_kv_to_select, sorted_kv_indices,
                                       num_q_to_select, sorted_q_indices,
                                         NBlock_KV, T2T=T2T)
    # print(f"sparse_info count_nonzero num: {torch.count_nonzero(sparse_info)}, ratio: {torch.count_nonzero(sparse_info) / sparse_info.numel()}")

    # print(f"sparse_kv_info count_nonzero num: {torch.count_nonzero(sparse_kv_info)}, ratio: {torch.count_nonzero(sparse_kv_info) / sparse_kv_info.numel()}")
    return sparse_kv_info, sparse_info

@triton.jit
def triton_fill_sparse_info_kernel(final_map, num_to_select, sorted_indices, NK: tl.constexpr, T2T: tl.constexpr = 0):
    b, h = tl.program_id(0), tl.program_id(1)
    B, H = tl.num_programs(0), tl.num_programs(1)
    cur_num_to_select = tl.load(num_to_select + b)
    cur_sorted_idx_ptr = sorted_indices + b * NK
    cur_final_map_ptr = final_map + b * H * NK + h * NK + T2T
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 0)

def fill_sparse_info_triton(sparse_info, num_to_select, sorted_indices, T2T=0):
    sparse_info = sparse_info.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, K = sparse_info.shape
    grid = (B, H)
    triton_fill_sparse_info_kernel[grid](sparse_info, num_to_select, sorted_indices, K, T2T=T2T)
    return sparse_info

@triton.jit
def triton_fill_sparse_kv_info_kernel(final_map, num_to_select, sorted_indices, NK: tl.constexpr, T2T: tl.constexpr = 0):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    cur_num_to_select = tl.load(num_to_select + b * H * Q + h * Q + q)
    cur_sorted_idx_ptr = sorted_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_final_map_ptr = final_map + b * H * (Q + T2T) * NK + h * (Q + T2T) * NK + (q + T2T) * NK 
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 0)

def fill_sparse_kv_info_triton(sparse_kv_info, num_to_select, sorted_indices, T2T=0):
    sparse_kv_info = sparse_kv_info.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, NBlock_Q, NBlock_KV = sparse_kv_info.shape
    grid = (B, H, NBlock_Q - T2T)
    triton_fill_sparse_kv_info_kernel[grid](sparse_kv_info, num_to_select, sorted_indices, NBlock_KV, T2T=T2T)
    return sparse_kv_info



if __name__ == "__main__":
    B, H, N, D = 1, 2, 5, 128
    nblock_q = N
    nblock_k = N
    pooled_score = torch.randn((B, H, N, N), device="cuda")
    T2T = 2
    T2T_kv = 2
    t2i = pooled_score[:, :, : T2T, T2T_kv: ].contiguous()
    t2i = torch.sum(t2i, dim=[1,2]).softmax(-1)
    sparse_info = torch.ones((B, H, nblock_q), dtype=torch.uint8, device="cuda")
    sorted_score = torch.sort(t2i, dim=-1, descending=False)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, K = cdf.shape
    cdfthreshd = torch.full((B,), float(0.2), device=pooled_score.device)
    
    cdfthreshd_ts = cdfthreshd.view(B, 1).contiguous()
    num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)
    sparse_info = fill_sparse_info_triton(sparse_info, num_to_select, sorted_score.indices, T2T=T2T)

    print('*' * 100, f"sparse_info part", '*' * 100)
    print("t2t", t2i)  
    print("sorted_score", sorted_score.values)
    print("sorted_score indices", sorted_score.indices)
    print("cdf", cdf)
    print("num_to_select", num_to_select)
    print("sparse_info", sparse_info)

    # sparse_kv_info --> for kv, B, H, nblock_q, nblock_k
    img2_ = pooled_score[:, :, T2T :, :].contiguous().softmax(-1)
    print(f"img2_ shape: {img2_.shape}")
    sparse_kv_info = torch.ones((B, H, nblock_q, nblock_k), dtype=torch.uint8, device="cuda")
    sorted_kv_score = torch.sort(img2_, dim=-1, descending=False)
    cdf_kv = torch.cumsum(sorted_kv_score.values, dim=-1)
    B, H, N_blockq, N_blockkv = cdf_kv.shape
    cdfthreshd = torch.full((H,), float(0.2), device=pooled_score.device)
    
    cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1).expand(B, -1, N_blockq,1).contiguous()
    num_to_select_kv = torch.searchsorted(cdf_kv, cdfthreshd_ts, right=True).squeeze(-1)
    sparse_kv_info = fill_sparse_kv_info_triton(sparse_kv_info, num_to_select_kv, sorted_kv_score.indices, T2T=T2T)
    # sparse_kv_info = sparse_kv_info.transpose(1, 2).contiguous().view(-1)
    print('*' * 100, f"sparse_kv_info part", '*' * 100)
    print("img2_", img2_)  
    print("sorted_kv_score", sorted_kv_score.values)
    print("sorted_kv_score indices", sorted_kv_score.indices)
    print("cdf", cdf_kv)
    print("num_to_select", num_to_select_kv)
    print("sparse_kv_info", sparse_kv_info)

    # sparse_kv_info --> for kv, B, H, nblock_q, nblock_k
    img2_all = pooled_score[:, :, :, :].contiguous().softmax(-1)
    sparse_kv_info_all = torch.ones((B, H, nblock_q, nblock_k), dtype=torch.uint8, device="cuda")
    sorted_kv_score_all = torch.sort(img2_all, dim=-1, descending=False)
    cdf_kv_all = torch.cumsum(sorted_kv_score_all.values, dim=-1)
    B, H, N_blockq, N_blockkv = cdf_kv_all.shape
    cdfthreshd_all = torch.full((H,), float(0.2), device=pooled_score.device)

    sparse_info_all = torch.ones((B, H, nblock_q), dtype=torch.uint8, device="cuda")

    cdfthreshd_ts_all = cdfthreshd_all.view(1, H, 1, 1).expand(B, -1, N_blockq, 1).contiguous()
    num_to_select_all = torch.searchsorted(cdf_kv_all, cdfthreshd_ts_all, right=True).squeeze(-1)
    sparse_kv_info_all, sparse_info_all = fill_flashomni_triton(sparse_kv_info_all, num_to_select_all, sorted_kv_score_all.indices, 
                                               sparse_info_all, num_to_select, sorted_score.indices, T2T=T2T)
    # sparse_kv_info_all = sparse_kv_info_all.transpose(1, 2).contiguous().view(-1)

    print(torch.equal(sparse_info, sparse_info_all))
    print(torch.equal(sparse_kv_info, sparse_kv_info_all))
    print(torch.equal(sparse_kv_info[:,:, T2T:], sparse_kv_info_all[:,:, T2T:]))
    print(torch.equal(sparse_kv_info[:,:, :T2T], torch.ones_like(sparse_kv_info[:,:, :T2T])))
    
