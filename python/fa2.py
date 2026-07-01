# A training-oriented FA2 with 4D mask & rightpad support
# Assum the 4D mask & score is static and do not requires grad

import os
import itertools
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune, set_autotune_inputs

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange
from functools import lru_cache

PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
}

###################### FWD ######################
@tilelang.jit(pass_configs=PASS_CFG)
def flash_attention_fwd_with_4d_mask(
    Q, K, V, O, LSE, Mask, Score,
    scale: float, has_mask: bool, has_score: bool, is_causal: bool,
    dtype: T.DType,
    BM: int=64,
    BN: int=64,
    Pipeline: int=2,
    Threads: int=128,
):
    B, Tq, Tk = T.dynamic('B, Tq, Tk')
    Hq, Hk, D = T.const('Hq, Hk, D')
    accum_dtype = T.float32
    scale *= 1.44269504
    head_group = Hq // Hk

    Q: T.Tensor[[B, Tq, Hq, D], dtype]
    K: T.Tensor[[B, Tk, Hk, D], dtype]
    V: T.Tensor[[B, Tk, Hk, D], dtype]
    O: T.Tensor[[B, Tq, Hq, D], dtype]
    LSE: T.Tensor[[B, Tq, Hq], accum_dtype]
    Mask: T.Tensor[[B, Hq, Tq, Tk], T.bool]
    Score: T.Tensor[[B, Hq, Tq, Tk], accum_dtype]

    with T.Kernel(T.cdiv(Tq, BM), Hq, B, threads=Threads) as (bidx, bidy, bidz):
        sQ = T.alloc_shared([BM, D], dtype)
        sK = T.alloc_shared([BN, D], dtype)
        sV = T.alloc_shared([BN, D], dtype)
        sP = T.alloc_shared([BM, BN], dtype)
        sO = T.alloc_shared([BM, D], dtype)

        rMask = T.alloc_fragment([BM, BN], T.bool)
        rScore = T.alloc_fragment([BM, BN], accum_dtype)

        rMax = T.alloc_fragment([BM], accum_dtype)
        rMax_tmp = T.alloc_fragment([BM], accum_dtype)
        rScale = T.alloc_fragment([BM], accum_dtype)
        rSum = T.alloc_fragment([BM], accum_dtype)
        rLogsum = T.alloc_fragment([BM], accum_dtype)
        rAcc = T.alloc_fragment([BM, D], accum_dtype)
        rPc = T.alloc_fragment([BM, BN], accum_dtype)

        query_head = bidy
        key_head = bidy // head_group
        T.fill(rMax, -T.infinity(accum_dtype))
        T.fill(rSum, 0.0)
        T.fill(rAcc, 0.0)

        q_start = bidx * BM
        q_end = q_start + BM

        T.copy(Q[bidz, q_start:q_end, query_head, :], sQ, disable_tma=True)
        kv_start = 0
        kv_end = T.min(Tk, q_end) if is_causal else Tk
        iter_num = T.max(0, T.cdiv(kv_end - kv_start, BN))

        for it in T.Pipelined(iter_num, num_stages=Pipeline):
            iter_kv_start = kv_start + it * BN
            iter_kv_end = iter_kv_start + BN

            T.copy(K[bidz, iter_kv_start:iter_kv_end, key_head, :], sK)
            T.copy(V[bidz, iter_kv_start:iter_kv_end, key_head, :], sV)
            T.clear(rPc)
            T.gemm(sQ, sK, rPc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            if has_score:
                T.copy(Score[bidz, query_head, q_start:q_end, iter_kv_start:iter_kv_end], rScore)
                for i, j in T.Parallel(BM, BN):
                    rPc[i, j] += rScore[i, j]

            if is_causal:
                for i, j in T.Parallel(BM, BN):
                    rPc[i, j] = T.if_then_else(
                        T.bitwise_and(iter_kv_start + j < kv_end, q_start + i >= iter_kv_start + j),
                        rPc[i, j] * scale,
                        -T.infinity(accum_dtype)
                    )
            else:
                for i, j in T.Parallel(BM, BN):
                    rPc[i, j] = T.if_then_else(
                        iter_kv_start + j < kv_end,
                        rPc[i, j] * scale,
                        -T.infinity(accum_dtype)
                    )
            
            if has_mask:
                T.copy(Mask[bidz, query_head, q_start:q_end, iter_kv_start:iter_kv_end], rMask)
                for i, j in T.Parallel(BM, BN):
                    rPc[i, j] = T.if_then_else(
                        T.bitwise_and(iter_kv_start + j < kv_end, rMask[i, j] == 1),
                        rPc[i, j],
                        -T.infinity(accum_dtype)
                    )
            
            T.fill(rMax_tmp, -T.infinity(accum_dtype))
            T.reduce_max(rPc, rMax_tmp, dim=-1, clear=False)
            for i in T.Parallel(BM):
                rMax_tmp[i] = T.max(rMax_tmp[i], rMax[i])
                rScale[i] = T.exp2(rMax[i] - rMax_tmp[i])
                rSum[i] *= rScale[i]
            for i, j in T.Parallel(BM, BN):
                rPc[i, j] = T.exp2(rPc[i, j] - rMax_tmp[i])
            T.reduce_sum(rPc, rSum, clear=False)
            for i, j in T.Parallel(BM, D):
                rAcc[i, j] *= rScale[i]
            
            T.copy(rPc, sP)
            T.gemm(sP, sV, rAcc, policy=T.GemmWarpPolicy.FullRow, clear_accum=False)
            T.copy(rMax_tmp, rMax)
        
        for i, j in T.Parallel(BM, D):
            rAcc[i, j] /= rSum[i]
        
        T.copy(rAcc, sO)
        T.copy(sO, O[bidz, q_start:q_end, query_head, :])
        for i in T.Parallel(BM):
            rLogsum[i] = T.log2(rSum[i]) + rMax[i]
        T.copy(rLogsum, LSE[bidz, q_start:q_end, query_head])

###################### BWD ######################
@tilelang.jit(pass_configs=PASS_CFG)
def bwd_preprocess(
    O, dO, Delta,
    dtype: T.DType,
    BM: int=64,
    BK: int=32,
    Threads: int=128,
):
    B, Tq = T.dynamic('B, Tq')
    Hq, D = T.const('Hq, D')
    accum_dtype = T.float32

    O: T.Tensor[[B, Tq, Hq, D], dtype]
    dO: T.Tensor[[B, Tq, Hq, D], dtype]
    Delta: T.Tensor[[B, Hq, Tq], accum_dtype]

    with T.Kernel(T.cdiv(Tq, BM), Hq, B, threads=Threads) as (bidx, bidy, bidz):
        rO = T.alloc_fragment([BM, BK], dtype)
        rdO = T.alloc_fragment([BM, BK], dtype)
        rAcc = T.alloc_fragment([BM, BK], accum_dtype)
        rDelta = T.alloc_fragment([BM], accum_dtype)

        q_start = bidx * BM
        q_end = q_start + BM
        T.clear(rAcc)
        for it in range(T.cdiv(D, BK)):
            T.copy(O[bidz, q_start:q_end, bidy, it * BK:(it + 1) * BK], rO)
            T.copy(dO[bidz, q_start:q_end, bidy, it * BK:(it + 1) * BK], rdO)
            for i, j in T.Parallel(BM, BK):
                rAcc[i, j] += rO[i, j] * rdO[i, j]
        T.reduce_sum(rAcc, rDelta)
        T.copy(rDelta, Delta[bidz, bidy, q_start:q_end])

@tilelang.jit(pass_configs=PASS_CFG)
def flash_attention_bwd_with_4d_mask_atomic(
    Q, K, V, dO, LSE, Delta, Mask, Score, dQ, dK, dV,
    scale: float, has_mask: bool, has_score: bool, is_causal: bool,
    dtype: T.DType,
    BM: int=64,
    BN: int=64,
    Pipeline: int=2,
    Threads: int=128,
):
    B, Tq, Tk = T.dynamic('B, Tq, Tk')
    Hq, Hk, D = T.const('Hq, Hk, D')
    accum_dtype = T.float32
    scale_new = scale * 1.44269504
    head_group = Hq // Hk

    Q: T.Tensor[[B, Tq, Hq, D], dtype]
    K: T.Tensor[[B, Tk, Hk, D], dtype]
    V: T.Tensor[[B, Tk, Hk, D], dtype]
    dO: T.Tensor[[B, Tq, Hq, D], dtype]
    LSE: T.Tensor[[B, Tq, Hq], accum_dtype]
    Delta: T.Tensor[[B, Hq, Tq], accum_dtype]
    Mask: T.Tensor[[B, Hq, Tq, Tk], T.bool]
    Score: T.Tensor[[B, Hq, Tq, Tk], accum_dtype]
    dQ: T.Tensor[[B, Tq, Hq, D], accum_dtype]
    dK: T.Tensor[[B, Tk, Hk, D], accum_dtype]
    dV: T.Tensor[[B, Tk, Hk, D], accum_dtype]

    with T.Kernel(T.cdiv(Tk, BM), Hq, B, threads=Threads) as (bidx, bidy, bidz):
        sQ = T.alloc_shared([BN, D], dtype)
        sK = T.alloc_shared([BM, D], dtype)
        sV = T.alloc_shared([BM, D], dtype)
        sO = T.alloc_shared([BN, D], dtype)
        sP = T.alloc_shared([BM, BN], dtype)
        sdS = T.alloc_shared([BM, BN], dtype)
        sdK = T.alloc_shared([BM, D], accum_dtype)
        sdV = T.alloc_shared([BM, D], accum_dtype)

        rMask = T.alloc_fragment([BN, BM], T.bool)
        rScore = T.alloc_fragment([BN, BM], accum_dtype)

        rPc = T.alloc_fragment([BM, BN], accum_dtype)
        rdS = T.alloc_fragment([BM, BN], accum_dtype)
        rdS_cast = T.alloc_fragment([BM, BN], dtype)
        rdQ = T.alloc_fragment([BN, D], accum_dtype)
        rdK = T.alloc_fragment([BM, D], accum_dtype)
        rdV = T.alloc_fragment([BM, D], accum_dtype)
        rLSE = T.alloc_fragment([BN], accum_dtype)
        rDelta = T.alloc_fragment([BN], accum_dtype)

        kv_start = bidx * BM
        kv_end = kv_start + BM
        query_head = bidy
        key_head = bidy // head_group

        T.copy(K[bidz, kv_start:kv_end, key_head, :], sK)
        T.copy(V[bidz, kv_start:kv_end, key_head, :], sV)

        T.clear(rdK)
        T.clear(rdV)

        q_start = kv_start if is_causal else 0
        q_end = Tq
        iter_num = T.max(0, T.cdiv(q_end - q_start, BN))
        for it in T.Pipelined(iter_num, num_stages=Pipeline):
            tile_q_start = q_start + it * BN
            tile_q_end = tile_q_start + BN
            T.copy(Q[bidz, tile_q_start:tile_q_end, bidy, :], sQ)
            T.clear(rPc)

            T.gemm(sK, sQ, rPc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            if has_score:
                T.copy(Score[bidz, bidy, tile_q_start:tile_q_end, kv_start:kv_end], rScore)
                for i, j in T.Parallel(BM, BN):
                    rPc[i, j] += rScore[j, i]

            T.copy(LSE[bidz, tile_q_start:tile_q_end, bidy], rLSE)
            for i, j in T.Parallel(BM, BN):
                rPc[i, j] = T.exp2(rPc[i, j] * scale_new - rLSE[j])
            
            if is_causal:
                for i, j in T.Parallel(BM, BN):
                    rPc[i, j] = T.if_then_else(
                        T.bitwise_and(tile_q_start + j < q_end, tile_q_start + j >= kv_start + i),
                        rPc[i, j], 0
                    )
            
            if has_mask:
                T.copy(Mask[bidz, bidy, tile_q_start:tile_q_end, kv_start:kv_end], rMask)
                for i, j in T.Parallel(BM, BN):
                    rPc[i, j] = T.if_then_else(
                        rMask[j, i] == 1,
                        rPc[i, j], 0
                    )
            
            T.copy(dO[bidz, tile_q_start:tile_q_end, bidy, :], sO)
            T.clear(rdS)
            T.gemm(sV, sO, rdS, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            T.copy(rPc, sP)
            T.gemm(sP, sO, rdV, policy=T.GemmWarpPolicy.FullRow)

            T.copy(Delta[bidz, bidy, tile_q_start:tile_q_end], rDelta)
            for i, j in T.Parallel(BM, BN):
                rdS_cast[i, j] = rPc[i, j] * (rdS[i, j] - rDelta[j]) * scale
            T.gemm(rdS_cast, sQ, rdK, policy=T.GemmWarpPolicy.FullRow)
            T.copy(rdS_cast, sdS)
            T.clear(rdQ)
            T.gemm(sdS, sK, rdQ, transpose_A=True)
            for i, j in T.Parallel(BN, D):
                T.atomic_add(dQ[bidz, tile_q_start + i, bidy, j], rdQ[i, j])

        T.copy(rdV, sdV)
        T.atomic_add(dV[bidz, kv_start:kv_end, key_head, :], sdV)
        T.copy(rdK, sdK)
        T.atomic_add(dK[bidz, kv_start:kv_end, key_head, :], sdK)


@tilelang.jit(pass_configs=PASS_CFG)
def flash_attention_bwd_with_4d_mask_reduce(
    Q, K, V, dO, LSE, Delta, Mask, Score, dQ, dK, dV,
    scale: float, has_mask: bool, has_score: bool, is_causal: bool,
    dtype: T.DType,
    BM: int=64,
    BN: int=64,
    Pipeline: int=2,
    Threads: int=128,
):
    B, Tq, Tk = T.dynamic('B, Tq, Tk')
    Hq, Hk, D = T.const('Hq, Hk, D')
    accum_dtype = T.float32
    scale_new = scale * 1.44269504
    head_group = Hq // Hk

    Q: T.Tensor[[B, Tq, Hq, D], dtype]
    K: T.Tensor[[B, Tk, Hk, D], dtype]
    V: T.Tensor[[B, Tk, Hk, D], dtype]
    dO: T.Tensor[[B, Tq, Hq, D], dtype]
    LSE: T.Tensor[[B, Tq, Hq], accum_dtype]
    Delta: T.Tensor[[B, Hq, Tq], accum_dtype]
    Mask: T.Tensor[[B, Hq, Tq, Tk], T.bool]
    Score: T.Tensor[[B, Hq, Tq, Tk], accum_dtype]
    dQ: T.Tensor[[B, Tq, Hq, D], accum_dtype]
    dK: T.Tensor[[head_group, B, Tk, Hk, D], dtype]
    dV: T.Tensor[[head_group, B, Tk, Hk, D], dtype]

    with T.Kernel(T.cdiv(Tk, BM), Hq, B, threads=Threads) as (bidx, bidy, bidz):
        sQ = T.alloc_shared([BN, D], dtype)
        sK = T.alloc_shared([BM, D], dtype)
        sV = T.alloc_shared([BM, D], dtype)
        sO = T.alloc_shared([BN, D], dtype)
        sP = T.alloc_shared([BM, BN], dtype)
        sdS = T.alloc_shared([BM, BN], dtype)
        sdK = T.alloc_shared([BM, D], dtype)
        sdV = T.alloc_shared([BM, D], dtype)

        rMask = T.alloc_fragment([BN, BM], T.bool)
        rScore = T.alloc_fragment([BN, BM], accum_dtype)

        rPc = T.alloc_fragment([BM, BN], accum_dtype)
        rdS = T.alloc_fragment([BM, BN], accum_dtype)
        rdS_cast = T.alloc_fragment([BM, BN], dtype)
        rdQ = T.alloc_fragment([BN, D], accum_dtype)
        rdK = T.alloc_fragment([BM, D], accum_dtype)
        rdV = T.alloc_fragment([BM, D], accum_dtype)
        rLSE = T.alloc_fragment([BN], accum_dtype)
        rDelta = T.alloc_fragment([BN], accum_dtype)

        kv_start = bidx * BM
        kv_end = kv_start + BM
        query_head = bidy
        key_head = bidy // head_group

        T.copy(K[bidz, kv_start:kv_end, key_head, :], sK)
        T.copy(V[bidz, kv_start:kv_end, key_head, :], sV)

        T.clear(rdK)
        T.clear(rdV)

        q_start = kv_start if is_causal else 0
        q_end = Tq
        iter_num = T.max(0, T.cdiv(q_end - q_start, BN))
        for it in T.Pipelined(iter_num, num_stages=Pipeline):
            tile_q_start = q_start + it * BN
            tile_q_end = tile_q_start + BN
            T.copy(Q[bidz, tile_q_start:tile_q_end, bidy, :], sQ)
            T.clear(rPc)

            T.gemm(sK, sQ, rPc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            if has_score:
                T.copy(Score[bidz, bidy, tile_q_start:tile_q_end, kv_start:kv_end], rScore)
                for i, j in T.Parallel(BM, BN):
                    rPc[i, j] += rScore[j, i]

            T.copy(LSE[bidz, tile_q_start:tile_q_end, bidy], rLSE)
            for i, j in T.Parallel(BM, BN):
                rPc[i, j] = T.exp2(rPc[i, j] * scale_new - rLSE[j])
            
            if is_causal:
                for i, j in T.Parallel(BM, BN):
                    rPc[i, j] = T.if_then_else(
                        T.bitwise_and(tile_q_start + j < q_end, tile_q_start + j >= kv_start + i),
                        rPc[i, j], 0
                    )
            
            if has_mask:
                T.copy(Mask[bidz, bidy, tile_q_start:tile_q_end, kv_start:kv_end], rMask)
                for i, j in T.Parallel(BM, BN):
                    rPc[i, j] = T.if_then_else(
                        rMask[j, i] == 1,
                        rPc[i, j], 0
                    )
            
            T.copy(dO[bidz, tile_q_start:tile_q_end, bidy, :], sO)
            T.clear(rdS)
            T.gemm(sV, sO, rdS, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            T.copy(rPc, sP)
            T.gemm(sP, sO, rdV, policy=T.GemmWarpPolicy.FullRow)

            T.copy(Delta[bidz, bidy, tile_q_start:tile_q_end], rDelta)
            for i, j in T.Parallel(BM, BN):
                rdS_cast[i, j] = rPc[i, j] * (rdS[i, j] - rDelta[j]) * scale
            T.gemm(rdS_cast, sQ, rdK, policy=T.GemmWarpPolicy.FullRow)
            T.copy(rdS_cast, sdS)
            T.clear(rdQ)
            T.gemm(sdS, sK, rdQ, transpose_A=True)
            for i, j in T.Parallel(BN, D):
                T.atomic_add(dQ[bidz, tile_q_start + i, bidy, j], rdQ[i, j])

        T.copy(rdV, sdV)
        T.copy(sdV, dV[bidy % head_group, bidz, kv_start:kv_end, key_head, :])
        T.copy(rdK, sdK)
        T.copy(sdK, dK[bidy % head_group, bidz, kv_start:kv_end, key_head, :])

@lru_cache
def get_cc() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor

def get_contiguous(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous(): return x.contiguous()
    return x

class _flash_attention_with_4d_mask(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        Mask: Optional[torch.Tensor]=None, Score: Optional[torch.Tensor]=None,
        scale: Optional[float]=None,
        is_causal: Optional[bool]=False,
        use_atomic_backward: Optional[bool]=True,
        bhsd_input: Optional[bool]=True,
    ):
        assert Q.dim() == 4 and K.dim() ==4 and V.dim() == 4
        if bhsd_input:
            Q = get_contiguous(Q.transpose(1, 2))
            K = get_contiguous(K.transpose(1, 2))
            V = get_contiguous(V.transpose(1, 2))

        B, Tq, Hq, D = Q.shape
        _, Tk, Hk, _ = K.shape
        if scale == None: scale = D ** -0.5

        # Tiling
        cc = get_cc()
        BM = 128
        BN = 32
        Pipeline = 2

        if cc in [80, 90, 100, 101, 102]:
            BM = 128
            BN = 64
            Pipeline = 2
        
        has_mask = False if Mask == None else True
        has_score = False if Score == None else True

        O = torch.empty([B, Tq, Hq, D], dtype=Q.dtype, device=Q.device)
        LSE = torch.empty([B, Tq, Hq], dtype=torch.float32, device=Q.device)
        flash_attention_fwd_with_4d_mask(
            Q, K, V, O, LSE, Mask, Score,
            scale, has_mask, has_score, is_causal, getattr(T, str(Q.dtype).split('.')[-1]),
            BM=BM, BN=BN, Pipeline=Pipeline,
        )

        ctx.save_for_backward(Q, K, V, O, LSE, Mask, Score)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.use_atomic_backward = use_atomic_backward
        ctx.has_mask = has_mask
        ctx.has_score = has_score
        ctx.bhsd_input = bhsd_input

        return O.transpose(1, 2) if bhsd_input else O
    
    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, K, V, O, LSE, Mask, Score = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        use_atomic_backward = ctx.use_atomic_backward
        has_mask = ctx.has_mask
        has_score = ctx.has_score
        bhsd_input = ctx.bhsd_input

        if bhsd_input: dO = get_contiguous(dO.transpose(1, 2))

        mask_arg = Mask if has_mask else None
        score_arg = Score if has_score else None

        B, Tq, Hq, D = Q.shape
        _, Tk, Hk, _ = K.shape

        Delta = torch.empty([B, Hq, Tq], dtype=torch.float32, device=Q.device)
        bwd_preprocess(
            O, dO, Delta,
            getattr(T, str(Q.dtype).split('.')[-1]),
            BM=128,
        )

        if use_atomic_backward:
            dQ = torch.zeros([B, Tq, Hq, D], dtype=torch.float32, device=Q.device)
            dK = torch.zeros([B, Tk, Hk, D], dtype=torch.float32, device=Q.device)
            dV = torch.zeros([B, Tk, Hk, D], dtype=torch.float32, device=Q.device)
            flash_attention_bwd_with_4d_mask_atomic(
                Q, K, V, dO, LSE, Delta, mask_arg, score_arg, dQ, dK, dV,
                scale, has_mask, has_score, is_causal, getattr(T, str(Q.dtype).split('.')[-1]),
                BM=128, BN=32, Pipeline=2, Threads=256,
            )
            dK = dK.to(Q.dtype)
            dV = dV.to(Q.dtype)
        
        else:
            kv_group = Hq // Hk
            dQ = torch.zeros([B, Tq, Hq, D], dtype=torch.float32, device=Q.device)
            dK = torch.empty([kv_group, B, Tk, Hk, D], dtype=Q.dtype, device=Q.device)
            dV = torch.empty([kv_group, B, Tk, Hk, D], dtype=Q.dtype, device=Q.device)
            flash_attention_bwd_with_4d_mask_reduce(
                Q, K, V, dO, LSE, Delta, mask_arg, score_arg, dQ, dK, dV,
                scale, has_mask, has_score, is_causal, getattr(T, str(Q.dtype).split('.')[-1]),
                BM=128, BN=32, Pipeline=2, Threads=256,
            )
            dK = dK.sum(0)
            dV = dV.sum(0)
        
        if bhsd_input:
            dQ = dQ.transpose(1, 2)
            dK = dK.transpose(1, 2)
            dV = dV.transpose(1, 2)

        return dQ, dK, dV, None, None, None, None, None, None

flash_attention_with_4d_mask = _flash_attention_with_4d_mask.apply