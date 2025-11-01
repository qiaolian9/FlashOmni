import pytest
import torch

import flashomni
import flashinfer
# import flash_attn
from flash_attn import flash_attn_func
from utils import get_qkvo, pretty_print_line, run_benchmark, check_all_close, get_qkvo_global_sparse
import argparse
from torch.nn import functional as F
import math
from flashomni import segment_packbits

torch.manual_seed(42)

def _compute_sparse_info_indptr(
    qo_indptr: torch.Tensor, num_qo_heads, sparse_block_size_for_q: int,
    device: torch.device = 'cuda'
) -> torch.Tensor:
    sparse_info_indptr = torch.empty_like(qo_indptr,device=device)
    sparse_info_indptr[0] = 0
    sparse_info_indptr[1:] = torch.cumsum(
        torch.ceil((qo_indptr[1:] - qo_indptr[:-1]) / sparse_block_size_for_q) * num_qo_heads,
        0,
    )
    sparse_info = torch.ones(sparse_info_indptr[-1], dtype=torch.uint8, device=device)
    
    return sparse_info, sparse_info_indptr

def _compute_sparse_kv_info_indptr(
    qo_indptr: torch.Tensor, kv_indptr: torch.Tensor, num_qo_heads, 
    sparse_block_size_for_q: int, sparse_block_size_for_kv: int,
    device: torch.device = 'cuda'
) -> torch.Tensor:
    sparse_kv_info_indptr = torch.empty_like(qo_indptr, device=device)
    sparse_kv_info_indptr[0] = 0
    sparse_kv_info_indptr[1:] = torch.cumsum(
        torch.ceil((qo_indptr[1:] - qo_indptr[:-1]) / sparse_block_size_for_q)
          * torch.ceil((kv_indptr[1:] - kv_indptr[:-1]) / sparse_block_size_for_kv) 
          * num_qo_heads,
        0,
    )
    sparse_kv_info = torch.ones(sparse_kv_info_indptr[-1], dtype=torch.uint8, device=device)

    return sparse_kv_info, sparse_kv_info_indptr


# un-fused naive attn
def unfused_standard_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

# un-fused naive attn
def unfused_standard_attn_sparse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sparse_Q: torch.Tensor, sparse_KV: torch.Tensor):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    mask = sparse_KV.repeat_interleave(sp_size, dim=-1).repeat_interleave(sp_size, dim=-2)  # shape (B, H, N)
    att[torch.where(mask == 0)] = float("-inf")
    att = F.softmax(att, dim=-1)
    mask = sparse_Q.repeat_interleave(sp_size, dim=-1)  # shape (B, H, N)
    att = att * mask.unsqueeze(-1).to(att.dtype)
    y = att @ v

    return y

def scaled_dot_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor = None):  
    
    y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
    return y

@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [False])
def flashinfer_with_ragged_kv(
    batch_size, qo_len, num_qo_heads, head_dim,
    kv_len,
    causal,
    pos_encoding_mode,
    logits_soft_cap,
):
    num_kv_heads = num_qo_heads
    kv_layout = "NHD"
    
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )

    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_len
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        q_data_type='float16',
    )
    return wrapper.run


def flashomni_mySparseKV(
    batch_size, qo_len, num_qo_heads, head_dim,
    kv_len,
    causal=False,
    pos_encoding_mode="NONE",
    logits_soft_cap=0.0,
    sparse_size = 128,
):
    num_kv_heads = num_qo_heads
    kv_layout = "NHD"
    
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )

    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_len
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    
    wrapper = flashomni.attention.BatchFlashOmniFAWithRaggedKVWrapper(
        workspace_buffer, kv_layout
    )

    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        q_data_type='float16',
        sparse_block_size_for_q= sparse_size,
        sparse_block_size_for_kv= sparse_size,
    )
    sparse_info, sparse_info_indptr = _compute_sparse_info_indptr(wrapper._qo_indptr_buf, 
                                                num_qo_heads, wrapper._sparse_block_size_for_q)
    
    sparse_kv_info, sparse_kv_info_indptr = _compute_sparse_kv_info_indptr(wrapper._qo_indptr_buf,
                                                wrapper._kv_indptr_buf, num_qo_heads, 
                                                wrapper._sparse_block_size_for_q, 
                                                wrapper._sparse_block_size_for_kv)
    return wrapper.run, sparse_info, sparse_info_indptr,sparse_kv_info, sparse_kv_info_indptr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", "--w", type=int, default=5)
    parser.add_argument("--iters", "--i", type=int, default=10)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    Bs = [1] 
    Hs = [24,] 
    Ns_tile = [1]
    text_token = 512
    Ns_q = [16384]
    # Ns_q = [4608, 6656, 8704, 16896]
    Ns_kv = [i + text_token for i in Ns_q]
    Ds = [128,] 
    Sparse_Q_percent = [0.0]
    Sparse_KV_percent = [0.8,]
    Sparse_size = [128]
    # batch_size, n_head, seq_len, head_dim (B,H,N,D)
    BHNDs = [(B, H, N_tile, N_kv, D, spQ, spKV, sp_size) for B in Bs for H in Hs for N_tile in Ns_tile for N_kv in Ns_kv for D in Ds for spQ in Sparse_Q_percent for spKV in Sparse_KV_percent for sp_size in Sparse_size]
    for (B, H, N_tile, N_kv, D, spQ, spKV, sp_size) in BHNDs:
        N = N_kv // N_tile
        max_tp = -1
        if N == 16896 and (B == 4 or B == 2) and H == 24:
            # skip this case, it is too large for my GPU
            continue
        print(f"Testing B={B}, H={H}, N={N}, N_kv={N_kv}, D={D}, sparse_Q_percent={spQ}, sparse_KV_percent={spKV}, sparse_size={sp_size}...")
        q, k, v, o, fq, fk, fv, fiq, fik, fiv  = get_qkvo(B, H, N, N_kv, D, dtype=torch.float16)
        sparse_info, sparse_Q, sparse_kv_info, sparse_KV = get_qkvo_global_sparse(B, H, N, N_kv, spq_Q=spQ, spq_KV=spKV, mode=1, sparse_size=sp_size, text_token = text_token)
        
        if args.check:
            # sparse attention mask
            attention_mask = sparse_KV.repeat_interleave(sp_size, dim=-1).repeat_interleave(sp_size, dim=-2).to(torch.bool)  # shape (B, H, N)
            attention_mask = attention_mask * sparse_Q.repeat_interleave(sp_size, dim=-1).unsqueeze(-1).to(torch.bool) 
            attention_mask = attention_mask[:, :, :N_kv, :N_kv]  # shape (B, H, N_kv, N_kv)

        sparse_kv_info = sparse_kv_info * sparse_info.unsqueeze(-1)
        print("sparse ratio", 1 - torch.count_nonzero(sparse_kv_info) / sparse_kv_info.numel())
        torch.cuda.synchronize()
        pretty_print_line()
        pretty_print_line(f"B={B}, H={H}, N={N}, D={D}, Warmup: {args.warmup}, Iters: {args.iters}")
        
        # run benchmarks
        #  =================   Naive MHA.  ====================.
        # if args.check:
        #     out_unfused,            throughput  = run_benchmark(unfused_standard_attn, q, k, v, tag="(unfused)", warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        # else:
        #     _,                      throughput  = run_benchmark(unfused_standard_attn, q, k, v, tag="(unfused)", warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        # max_tp                      = max(max_tp, throughput)

        #  =================   scaled_dot_attn.  ====================.
        if args.check:
            out_scaled_dot,         throughput  = run_benchmark(scaled_dot_attn, q, k, v, tag="(scaled_dot)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        else:
            _,                      throughput  = run_benchmark(scaled_dot_attn, q, k, v, tag="(scaled_dot)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        max_tp                      = max(max_tp, throughput)

        #  =================   unfused_standard_attn_sparse.  ====================.
        # if args.check:
        #     out_unfused_masked,     throughput  = run_benchmark(unfused_standard_attn_sparse, q, k, v, sparse_info= sparse_Q, sparse_kv_info=sparse_KV,  tag="(unfused Masked)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        # else:
        #     _,                      throughput  = run_benchmark(unfused_standard_attn_sparse, q, k, v, sparse_info= sparse_Q, sparse_kv_info=sparse_KV,  tag="(unfused Masked)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        # max_tp                      = max(max_tp, throughput)

        #  =================   scaled_dot_attn with sparse mask.  ====================.
        if args.check:
            out_scaled_dot_masked,  throughput  = run_benchmark(scaled_dot_attn, q, k, v, attention_mask=attention_mask, tag="(scaled_dot Masked)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        
        #  =================   flash attention.  ====================.
        if args.check:
            out_flash,              throughput  = run_benchmark(flash_attn_func, fq, fk, fv, tag="(flash)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        else:
            _,                      throughput  = run_benchmark(flash_attn_func, fq, fk, fv, tag="(flash)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        max_tp                      = max(max_tp, throughput)

        #  =================   flashinfer attention.  ====================.
        wrapper                                 = flashinfer_with_ragged_kv(B, N, H, D, N_kv, causal=False, pos_encoding_mode="NONE", logits_soft_cap=0.0)
        if args.check:
            out_flashinfer,         throughput  = run_benchmark(wrapper, fiq, fik, fiv, tag="(flashinfer)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
            out_flashinfer                      = out_flashinfer.view(B, N, H, D).transpose(1, 2).contiguous()
        else:
            _,                      throughput  = run_benchmark(wrapper, fiq, fik, fiv, tag="(flashinfer)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        max_tp                      = max(max_tp, throughput)
        

        #  =================   flashomni attention.  ====================.
        wrapper, fake_sparse_info, sparse_info_indptr, fake_sparse_kv_info, sparse_kv_info_indptr  = flashomni_mySparseKV(B, N, H, D, N_kv, causal=False, pos_encoding_mode="NONE", logits_soft_cap=0.0, sparse_size=sp_size)
        assert sparse_info.contiguous().view(-1).shape[0] == sparse_info_indptr[-1], "Error: sparse_info size does not match sparse_info_indptr"
        assert sparse_kv_info.contiguous().view(-1).shape[0] == sparse_kv_info_indptr[-1], "Error: sparse_kv_info size does not match sparse_kv_info_indptr"
        packed_sparse_info, sparse_info_indptr = segment_packbits(sparse_info.contiguous().view(-1), sparse_info_indptr, bitorder="little")
        packed_sparse_kv_info, sparse_kv_info_indptr = segment_packbits(sparse_kv_info.contiguous().view(-1), sparse_kv_info_indptr, bitorder="little")
        if args.check:
            out_flashomni_sparse,   throughput  = run_benchmark(wrapper, fiq, fik, fiv, sparse_info=packed_sparse_info, 
                                                            sparse_kv_info=packed_sparse_kv_info, 
                                                            sparse_info_indptr=sparse_info_indptr, 
                                                            sparse_kv_info_indptr=sparse_kv_info_indptr,
                                                            tag="(sparse)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
            out_flashomni_sparse                = out_flashomni_sparse.view(B, N, H, D).transpose(1, 2).contiguous()
        else:
            _,                      throughput  = run_benchmark(wrapper, fiq, fik, fiv, sparse_info=packed_sparse_info, 
                                                            sparse_kv_info=packed_sparse_kv_info, 
                                                            sparse_info_indptr=sparse_info_indptr, 
                                                            sparse_kv_info_indptr=sparse_kv_info_indptr,
                                                            tag="(sparse)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        max_tp                      = max(max_tp, throughput)

        # if args.check:
        #     out_flashomni_sparse_full,   throughput  = run_benchmark(wrapper, fiq, fik, fiv, sparse_info=packed_sparse_info, 
        #                                                     sparse_kv_info=packed_sparse_kv_info, 
        #                                                     sparse_info_indptr=sparse_info_indptr, 
        #                                                     sparse_kv_info_indptr=sparse_kv_info_indptr,
        #                                                     is_full=True,
        #                                                     tag="(sparse_full)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        #     out_flashomni_sparse_full                = out_flashomni_sparse_full.view(B, N, H, D).transpose(1, 2).contiguous()
        # else:
        #     _,                      throughput  = run_benchmark(wrapper, fiq, fik, fiv, sparse_info=packed_sparse_info, 
        #                                                     sparse_kv_info=packed_sparse_kv_info, 
        #                                                     sparse_info_indptr=sparse_info_indptr, 
        #                                                     sparse_kv_info_indptr=sparse_kv_info_indptr,
        #                                                     is_full=True,
        #                                                     tag="(sparse_full)",  warmup=args.warmup, iters=args.iters, MAX_Throughput=max_tp)
        # max_tp                      = max(max_tp, throughput)
        
        pretty_print_line()
        torch.cuda.synchronize()
        if args.check:
            pretty_print_line()
            # check_all_close(out_flash, out_unfused,                          tag1="flash",              tag2="unfused")
            check_all_close(out_flash, out_flashinfer,                       tag1="flash",              tag2="flashinfer")
            check_all_close(out_flash, out_scaled_dot,                       tag1="flash",              tag2="scaled_dot")
            check_all_close(out_scaled_dot, out_flashinfer,                  tag1="scaled_dot",         tag2="flashinfer",          is_flash=False)
            # check_all_close(out_unfused_masked, out_flashomni_sparse,        tag1="unfused",            tag2="flashomni",           is_flash=False)
            check_all_close(out_scaled_dot_masked, out_flashomni_sparse,     tag1="scaled_dot_masked",  tag2="flashomni",           is_flash=False)
            
            # ============= compare with full attention output ==============
            # check_all_close(out_flash, out_flashomni_sparse_full,     tag1="flash",  tag2="flashomni_full")
            # check_all_close(out_scaled_dot, out_flashomni_sparse_full,                  tag1="scaled_dot",         tag2="flashomni_full",          is_flash=False)
            # check_all_close(out_flash, out_flashomni_sparse_full,       tag1="flash",            tag2="flashomni_full")
            pretty_print_line()
