import torch
import pytest
import time
from typing import Optional
import math

def get_qkvo_global_sparse(B, H, N, N_kv, spq_Q=0.25, spq_KV=0.25, mode=1, sparse_size = 128, text_token = 512):
    """
    Generates a sparse index tensor (mask) for Q with random global sparsification,
    and a corresponding global index_info tensor.

    Args:
        B (int): Batch size.
        H (int): Number of heads.
        N (int): Sequence length.
        spq_Q (float): Global sparsity ratio for Q indices (0.0 to 1.0).

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The globally sparse Q index tensor (mask, shape B x H x sparse_num).
            - torch.Tensor: The global index_info tensor (shape B*H*sparse_num).
    """
    # Determine the size of the sparse dimension per head.
    # This sparse_num corresponds to N / 512 as per original logic.
    sparse_num = math.ceil(N / sparse_size)
    sparse_kv_num = math.ceil(N_kv / sparse_size)
    
    # Handle case where sparse_num is 0
    assert(sparse_num > 0), "sparse_num must be greater than 0"


    # Total number of elements in the sparse_Q_index tensor across all B and H
    total_elements = B * H * sparse_num
    total_elements_for_kv = B * H * sparse_num * sparse_kv_num

    # --- Generate the sparse mask (sparse_Q_index) ---
    # Initialize mask with ones (shape B x H x sparse_num)
    sparse_Q_index = torch.ones((B, H, sparse_num), device="cuda", dtype=torch.uint8)
    sparse_KV_index = torch.ones((B, H, sparse_num, sparse_kv_num), device="cuda", dtype=torch.uint8)
    
    # Calculate the number of elements to set to zero based on global sparsity ratio
    num_to_zero = int(B * H * (sparse_num) * spq_Q)
    num_to_zero_kv = int(B * H * sparse_num * (sparse_kv_num) * spq_KV)

    if num_to_zero > 0:
        # Generate flat random indices to zero out from the flattened view
        flat_indices_to_zero = torch.randperm(total_elements, device="cuda")[:num_to_zero]

        # Convert flattened indices back to 3D indices (batch, head, sparse_index)
        # Calculate the strides for each dimension in the flattened view
        stride_h = sparse_num # Elements per head slice
        stride_b = H * stride_h # Elements per batch slice

        # Calculate the 3D indices using integer division and modulo
        batch_indices = flat_indices_to_zero // stride_b
        head_indices = (flat_indices_to_zero % stride_b) // stride_h
        sparse_indices = flat_indices_to_zero % stride_h

        # Use advanced indexing to set the selected elements in the mask to 0
        sparse_Q_index[batch_indices, head_indices, sparse_indices] = 0
    
    if num_to_zero_kv > 0:
        # Generate flat random indices to zero out from the flattened view
        flat_indices_to_zero = torch.randperm(total_elements_for_kv, device="cuda")[:num_to_zero_kv]
        # print(flat_indices_to_zero)
        # Convert flattened indices back to 3D indices (batch, head, sparse_index)
        # Calculate the strides for each dimension in the flattened view
        stride_n = sparse_kv_num # Elements per head slice
        stride_h = sparse_num * stride_n # Elements per head slice
        stride_b = H * stride_h # Elements per batch slice

        # Calculate the 3D indices using integer division and modulo
        batch_indices = flat_indices_to_zero // stride_b
        head_indices = (flat_indices_to_zero % stride_b) // stride_h
        Q_indices = (flat_indices_to_zero % stride_h) // stride_n
        sparse_indices = flat_indices_to_zero % stride_n

        # Use advanced indexing to set the selected elements in the mask to 0
        # sparse_Q_index[batch_indices, head_indices, sparse_indices] = 0
        sparse_KV_index[batch_indices, head_indices, Q_indices, sparse_indices] = 0

    # print(num_to_zero)
    sparse_info = sparse_Q_index.transpose(1, 2).contiguous().view(-1, H)
    sparse_kv_info = sparse_KV_index.transpose(1, 2).contiguous().view(-1, H, sparse_kv_num)
    return sparse_info, sparse_Q_index, sparse_kv_info, sparse_KV_index

def get_qkvo(B, H, N_q, N_kv, D, dtype=torch.bfloat16):
    q = torch.randn((B, H, N_q, D), dtype=dtype, device="cuda")
    k = torch.randn((B, H, N_kv, D), dtype=dtype, device="cuda")
    v = torch.randn((B, H, N_kv, D), dtype=dtype, device="cuda")
    o = torch.zeros(B, H, N_q, D, device="cuda", dtype=dtype).contiguous()
    # transpose (H,N) -> (N,H) for FA2.
    fq = q.transpose(1,   2).contiguous()
    fk = k.transpose(1,   2).contiguous()
    fv = v.transpose(1,   2).contiguous()
    
    # transpose (H,N) -> (N,H) for FA2.
    fiq = q.transpose(1,   2).contiguous().reshape(B * N_q, H, D)
    fik = k.transpose(1,   2).contiguous().reshape(B * N_kv, H, D)
    fiv = v.transpose(1,   2).contiguous().reshape(B * N_kv, H, D)

    return q, k, v, o, fq, fk, fv, fiq, fik, fiv

def run_benchmark(perf_func: callable, 
                  q: torch.Tensor, 
                  k: torch.Tensor, 
                  v: torch.Tensor,
                  sparse_info: Optional[torch.Tensor] = None,
                  sparse_kv_info: Optional[torch.Tensor] = None,
                  sparse_info_indptr: Optional[torch.Tensor] = None,
                  sparse_kv_info_indptr: Optional[torch.Tensor] = None,
                  is_full: bool = False,
                  attention_mask: Optional[torch.Tensor] = None,
                  tag: str = None, 
                  out: Optional[torch.Tensor] = None, 
                  warmup: int = 5, 
                  iters: int = 10,
                  MAX_Throughput: float = -1.0
                  ):
    
    with torch.no_grad():
        for i in range(warmup):
            if sparse_info is not None and sparse_kv_info is not None:
                if sparse_info_indptr is not None and sparse_kv_info_indptr is not None:
                    _ = perf_func(q, k, v, sparse_info, sparse_kv_info, sparse_info_indptr, sparse_kv_info_indptr, is_full)
                else:
                    _ = perf_func(q, k, v, sparse_info, sparse_kv_info)
            elif attention_mask is not None:
                _ = perf_func(q, k, v, attention_mask = attention_mask)
            else:
                _ = perf_func(q, k, v)
        
        torch.cuda.synchronize()
        
        start = time.time()
        for i in range(iters):
            if sparse_info is not None and sparse_kv_info is not None:
                if sparse_info_indptr is not None and sparse_kv_info_indptr is not None:
                    out = perf_func(q, k, v, sparse_info, sparse_kv_info, sparse_info_indptr, sparse_kv_info_indptr, is_full)
                else:
                    out = perf_func(q, k, v, sparse_info, sparse_kv_info)
            elif attention_mask is not None:
                out = perf_func(q, k, v, attention_mask = attention_mask)
            else:
                out = perf_func(q, k, v)
        torch.cuda.synchronize()
        end = time.time()
        total_secs = (end - start)
        total_time = (end - start) * 1000 # ms
        mean_time = total_time / iters
        mean_secs = total_secs / iters
        out_info = f"{tag}"

        Throughput = 1 / mean_secs

        out_info = f"{tag}"
        if out.dtype == torch.bfloat16:
            out_val = 'none'
        else:
            out_val_first = out.flatten()[:3].detach().cpu().numpy().tolist()
            out_val_last = out.flatten()[-3:].detach().cpu().numpy().tolist()
            out_val_first = [round(v, 8) for v in out_val_first]
            out_val_last = [round(v, 8) for v in out_val_last]
            out_val = out_val_first[:2]
            out_val.append(out_val_last[-1])
            out_val = [f"{v:<12}" for v in out_val]
        # caculate Throughput improved.
        if Throughput > MAX_Throughput:
            if MAX_Throughput > 0:
                improve = ((Throughput - MAX_Throughput) / MAX_Throughput) * 100
                improve = round(improve, 2)
            else:
                improve = 0
            print(f"{out_info:>50}: {out_val}, time:{str(mean_time)[:8]}ms, "
                f"ThroughtPut:{Throughput:<6.2f} samples/s (w/ improve {improve}%)")
        else:
            neg_improve = ((Throughput - MAX_Throughput) / MAX_Throughput) * 100
            neg_improve = round(neg_improve, 2)
            print(f"{out_info:>50}: {out_val}, time:{str(mean_time)[:8]}ms, "
                    f"ThroughtPut:{Throughput:<6.2f} samples/s (w/ -improve {neg_improve}%)")
                
        time.sleep(0.05)
        torch.cuda.synchronize()
        return out.clone(), Throughput


def run_benchmark_attention_block(perf_func: callable, 
                  norm_hidden_states: torch.Tensor, 
                  norm_encoder_hidden_states: torch.Tensor,
                  attention_mask: Optional[torch.Tensor] = None,
                  tag: str = None, 
                  out: Optional[torch.Tensor] = None, 
                  warmup: int = 5, 
                  iters: int = 10,
                  MAX_Throughput: float = -1.0
                  ):
    
    with torch.no_grad():
        for i in range(warmup):
            if attention_mask is not None:
                _ = perf_func(norm_hidden_states, norm_encoder_hidden_states, attention_mask=attention_mask)
            else:
                _ = perf_func(norm_hidden_states, norm_encoder_hidden_states)
        
        torch.cuda.synchronize()
        # iters
        start = time.time()
        for i in range(iters):
            if attention_mask is not None:
                out = perf_func(norm_hidden_states, norm_encoder_hidden_states, attention_mask=attention_mask)
            else:
                out = perf_func(norm_hidden_states, norm_encoder_hidden_states)

        torch.cuda.synchronize()
        end = time.time()
        total_secs = (end - start)
        total_time = (end - start) * 1000 # ms
        mean_time = total_time / iters
        mean_secs = total_secs / iters
        out_info = f"{tag}"

        Throughput = 1 / mean_secs

        out_info = f"{tag}"
        # caculate Throughput improved.
        if Throughput > MAX_Throughput:
            if MAX_Throughput > 0:
                improve = ((Throughput - MAX_Throughput) / MAX_Throughput) * 100
                improve = round(improve, 2)
            else:
                improve = 0
            print(f"{out_info:>50}, time:{str(mean_time)[:8]}ms, "
                f"ThroughtPut:{Throughput:<6.2f} samples/s (w/ improve {improve}%)")
        else:
            neg_improve = ((Throughput - MAX_Throughput) / MAX_Throughput) * 100
            neg_improve = round(neg_improve, 2)
            print(f"{out_info:>50}, time:{str(mean_time)[:8]}ms, "
                    f"ThroughtPut:{Throughput:<6.2f} samples/s (w/ -improve {neg_improve}%)")
                
        time.sleep(0.05)
        torch.cuda.synchronize()
        return out, Throughput


def run_benchmark_transformer_block(perf_func: callable, 
                  norm_hidden_states: torch.Tensor, 
                  norm_encoder_hidden_states: torch.Tensor,
                  temb: torch.Tensor,
                  attention_mask: Optional[torch.Tensor] = None,
                  tag: str = None, 
                  out: Optional[torch.Tensor] = None, 
                  warmup: int = 5, 
                  iters: int = 10,
                  MAX_Throughput: float = -1.0
                  ):
    
    with torch.no_grad():
        for i in range(warmup):
            if attention_mask is not None:
                _ = perf_func(norm_hidden_states, norm_encoder_hidden_states, temb = temb, attention_mask=attention_mask)
            else:
                _ = perf_func(norm_hidden_states, norm_encoder_hidden_states, temb = temb)
        
        torch.cuda.synchronize()
        # iters
        start = time.time()
        for i in range(iters):
            if attention_mask is not None:
                out = perf_func(norm_hidden_states, norm_encoder_hidden_states, temb = temb, attention_mask=attention_mask)
            else:
                out = perf_func(norm_hidden_states, norm_encoder_hidden_states, temb = temb)

        torch.cuda.synchronize()
        end = time.time()
        total_secs = (end - start)
        total_time = (end - start) * 1000 # ms
        mean_time = total_time / iters
        mean_secs = total_secs / iters
        out_info = f"{tag}"

        Throughput = 1 / mean_secs

        out_info = f"{tag}"
        # caculate Throughput improved.
        if Throughput > MAX_Throughput:
            if MAX_Throughput > 0:
                improve = ((Throughput - MAX_Throughput) / MAX_Throughput) * 100
                improve = round(improve, 2)
            else:
                improve = 0
            print(f"{out_info:>50}, time:{str(mean_time)[:8]}ms, "
                f"ThroughtPut:{Throughput:<6.2f} samples/s (w/ improve {improve}%)")
        else:
            neg_improve = ((Throughput - MAX_Throughput) / MAX_Throughput) * 100
            neg_improve = round(neg_improve, 2)
            print(f"{out_info:>50}, time:{str(mean_time)[:8]}ms, "
                    f"ThroughtPut:{Throughput:<6.2f} samples/s (w/ -improve {neg_improve}%)")
                
        time.sleep(0.05)
        torch.cuda.synchronize()
        return out, Throughput


def run_benchmark_singletransformer_block(perf_func: callable, 
                  norm_hidden_states: torch.Tensor, 
                  temb: torch.Tensor,
                  attention_mask: Optional[torch.Tensor] = None,
                  tag: str = None, 
                  out: Optional[torch.Tensor] = None, 
                  warmup: int = 5, 
                  iters: int = 10,
                  MAX_Throughput: float = -1.0
                  ):
    
    with torch.no_grad():
        for i in range(warmup):
            if attention_mask is not None:
                _ = perf_func(norm_hidden_states, temb = temb, attention_mask=attention_mask)
            else:
                _ = perf_func(norm_hidden_states, temb = temb)
        
        torch.cuda.synchronize()
        # iters
        start = time.time()
        for i in range(iters):
            if attention_mask is not None:
                out = perf_func(norm_hidden_states, temb = temb, attention_mask=attention_mask)
            else:
                out = perf_func(norm_hidden_states, temb = temb)

        torch.cuda.synchronize()
        end = time.time()
        total_secs = (end - start)
        total_time = (end - start) * 1000 # ms
        mean_time = total_time / iters
        mean_secs = total_secs / iters
        out_info = f"{tag}"

        Throughput = 1 / mean_secs

        out_info = f"{tag}"
        # caculate Throughput improved.
        if Throughput > MAX_Throughput:
            if MAX_Throughput > 0:
                improve = ((Throughput - MAX_Throughput) / MAX_Throughput) * 100
                improve = round(improve, 2)
            else:
                improve = 0
            print(f"{out_info:>50}, time:{str(mean_time)[:8]}ms, "
                f"ThroughtPut:{Throughput:<6.2f} samples/s (w/ improve {improve}%)")
        else:
            neg_improve = ((Throughput - MAX_Throughput) / MAX_Throughput) * 100
            neg_improve = round(neg_improve, 2)
            print(f"{out_info:>50}, time:{str(mean_time)[:8]}ms, "
                    f"ThroughtPut:{Throughput:<6.2f} samples/s (w/ -improve {neg_improve}%)")
                
        time.sleep(0.05)
        torch.cuda.synchronize()
        return out, Throughput
    
def run_benchmark_linear(perf_func: callable, 
                  x: torch.Tensor, 
                  w: torch.Tensor = None, 
                  num_qo_heads: int = None,
                  sparse_info: Optional[torch.Tensor] = None,
                  sparse_info_indptr: Optional[torch.Tensor] = None,
                  tag: str = None, 
                  out: Optional[torch.Tensor] = None, 
                  warmup: int = 5, 
                  iters: int = 10,
                  MAX_Throughput: float = -1.0,
                  bias: Optional[torch.Tensor] = None,
                  ):
    
    with torch.no_grad():
        for i in range(warmup):
            if sparse_info is not None and sparse_info_indptr is not None:
                 _ = perf_func(x, w, num_qo_heads, sparse_info, sparse_info_indptr, True, bias)
            else:
                _ = perf_func(x)
        
        torch.cuda.synchronize()
        
        start = time.time()
        for i in range(iters):
            if sparse_info is not None and sparse_info_indptr is not None:
                out = perf_func(x, w, num_qo_heads, sparse_info, sparse_info_indptr, True, bias)
            else:
                out = perf_func(x)
        torch.cuda.synchronize()
        end = time.time()
        total_secs = (end - start)
        total_time = (end - start) * 1000 # ms
        mean_time = total_time / iters
        mean_secs = total_secs / iters
        out_info = f"{tag}"

        Throughput = 1 / mean_secs

        out_info = f"{tag}"
        # caculate Throughput improved.
        if Throughput > MAX_Throughput:
            if MAX_Throughput > 0:
                improve = ((Throughput - MAX_Throughput) / MAX_Throughput) * 100
                improve = round(improve, 2)
            else:
                improve = 0
            print(f"{out_info:>50}: time:{str(mean_time)[:8]}ms, "
                f"ThroughtPut:{Throughput:<6.2f} samples/s (w/ improve {improve}%)")
        else:
            neg_improve = ((Throughput - MAX_Throughput) / MAX_Throughput) * 100
            neg_improve = round(neg_improve, 2)
            print(f"{out_info:>50}: time:{str(mean_time)[:8]}ms, "
                    f"ThroughtPut:{Throughput:<6.2f} samples/s (w/ -improve {neg_improve}%)")
                
        time.sleep(0.05)
        torch.cuda.synchronize()
        return out.clone(), Throughput



def check_all_close(out_flash: torch.Tensor, out_others: torch.Tensor, 
                    tag1: str = "torch",
                    tag2: str = "flashomni", 
                    is_flash: bool = True,
                    is_single_kernel: bool = True,
                    atol = 1e-3):
    if any((out_flash is None, out_others is None)):
        return
    if is_flash:
        out_flash = out_flash.transpose(1, 2)

    diff = torch.abs(out_flash - out_others)
    all_close = str(torch.allclose(out_flash, out_others, atol=atol))
    if is_single_kernel:
        tag_info = f"{tag1} vs {tag2}, all close: {all_close:<6}"
    else:
        tag_info = f"{tag1} vs {tag2}"
    pretty_print_line(
        f"{tag_info},"
        f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
        f"mean diff: {diff.mean().item():.6f}"
    )

def pretty_print_line(m: str = "", sep: str = "-", width: int = 150):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)