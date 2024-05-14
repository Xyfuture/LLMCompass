from dataclasses import dataclass

from software_model.operators import (
    Operator,
    Reshape,
    Concat,
    Transpose,
)
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU

from software_model.utils import Tensor, DataType
from software_model.communication_primitives import AllReduceMultiPCB
from math import ceil
from typing import List, Optional
from hardware_model.system import System


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    seq_length: int = 1024
    batch_size: int = 128


# class TransformerBlockInitComputationTP(Operator):
#     def __init__(self, model_args: ModelArgs, d_model, n_heads, device_count, data_type: DataType):
#         super().__init__(0, 0, 0, 0, data_type)
#         # self.d_model = d_model
#         # self.n_heads = n_heads
#         self.device_count = device_count
#
#         self.d_model = model_args.dim
#         self.n_heads = model_args.n_heads
#
#         # ffn_dim
#         hidden_dim = model_args.dim
#         hidden_dim = int(2 * hidden_dim / 3)
#         # custom dim factor multiplier
#         if model_args.ffn_dim_multiplier is not None:
#             hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
#         hidden_dim = model_args.multiple_of * ((hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of)
#
#         self.ffn_dim = hidden_dim
#
#         # parameters per device
#         d = d_model
#         self.Wq = Tensor([d, d // device_count], data_type)
#         self.Wk = Tensor([d, d // device_count], data_type)
#         self.Wv = Tensor([d, d // device_count], data_type)
#         self.Wo = Tensor([d // device_count, d], data_type)
#
#         self.W1 = Tensor([d, self.ffn_dim // device_count], data_type)
#         self.W2 = Tensor([d, self.ffn_dim // device_count], data_type)
#         self.W3 = Tensor([self.ffn_dim // device_count, d], data_type)
#         # operators per device
#         # # multi-head attention
#         self.Q_proj = Matmul(data_type)
#         self.K_proj = Matmul(data_type)
#         self.V_proj = Matmul(data_type)
#         self.Q_reshape = Reshape(data_type)
#         self.K_reshape = Reshape(data_type)
#         self.V_reshape = Reshape(data_type)
#         self.Q_transpose = Transpose(data_type)
#         self.K_transpose = Transpose(data_type)
#         self.V_transpose = Transpose(data_type)
#         self.Q_mul_K = BatchedMatmul(data_type)
#         self.A_softmax = Softmax(data_type)
#         self.A_mul_V = BatchedMatmul(data_type)
#         self.H_transpose = Transpose(data_type)
#         self.H_reshape = Reshape(data_type)
#         self.H_matmul_o = Matmul(data_type)
#         self.layer_norm_o = LayerNorm(data_type)
#         self.allreduce_mha = AllReduceMultiPCB(data_type)
#         # # feed-forward network
#         self.H_matmul_1 = Matmul(data_type)
#         self.H_gelu = GeLU(data_type)
#         self.H_matmul_2 = Matmul(data_type)
#         # TODO element wise mul
#         self.H_matmul_3 = Matmul(data_type)
#
#
#         self.layer_norm_1 = LayerNorm(data_type)
#         self.allreduce_ffn = AllReduceMultiPCB(data_type)
#
#     def __call__(self, X: Tensor) -> Tensor:
#         # b: batch size
#         # s: sequence length
#         # d: hidden dimension
#         # d_h: dimension per head
#
#         b, s, d = X.shape
#         assert d == self.d_model
#         h = self.n_heads
#         dev_cnt = self.device_count
#         d_h = d // h
#
#         # multi-head attention
#         Q = self.Q_proj(X, self.Wq)  # [b, s, d / dev_cnt]
#         assert Q.shape == [b, s, d // dev_cnt]
#         K = self.K_proj(X, self.Wk)  # [b, s, d / dev_cnt]
#         V = self.V_proj(X, self.Wv)  # [b, s, d / dev_cnt]
#         Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
#         K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
#         V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])
#         Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
#         assert Q_T.shape == [b, h // dev_cnt, s, d_h]
#         K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, s]
#         assert K_T.shape == [b, h // dev_cnt, d_h, s]
#         V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
#         assert V_T.shape == [b, h // dev_cnt, s, d_h]
#         A = self.Q_mul_K(Q_T, K_T)  # [b, h / dev_cnt, s, s]
#         assert A.shape == [b, h // dev_cnt, s, s]
#         A_prob = self.A_softmax(A)
#         H = self.A_mul_V(A_prob, V_T)  # [b, h / dev_cnt, s, d_h]
#         assert H.shape == [b, h // dev_cnt, s, d_h]
#         H = self.H_transpose(H, [0, 2, 1, 3])  # [b, s, h / dev_cnt, d_h]
#         assert H.shape == [b, s, h // dev_cnt, d_h]
#         H = self.H_reshape(H, [b, s, d // dev_cnt])
#         assert H.shape == [b, s, d // dev_cnt]
#         H0 = self.H_matmul_o(H, self.Wo)  # [b, s, d]
#         assert H0.shape == [b, s, d]
#         H0 = self.layer_norm_o(H0)
#         assert H0.shape == [b, s, d]
#         if dev_cnt > 1:
#             H0 = self.allreduce_mha(H0)
#
#         # feed-forward network
#         H1 = self.H_matmul_1(H0, self.W1)  # [b, s, 4 * d / dev_cnt]
#         assert H1.shape == [b, s, self.ffn_dim // dev_cnt]
#         H1 = self.H_gelu(H1)
#
#         H2 = self.H_matmul_2(H0,self.W2)
#         # TODO element dot
#
#         H3 = self.H_matmul_3(H1, self.W3)  # [b, s, d]
#         assert H3.shape == [b, s, d]
#         H3 = self.layer_norm_1(H3)
#         if dev_cnt > 1:
#             H3 = self.allreduce_ffn(H3)
#
#         assert H3.shape == [b, s, d]
#         return H3
#
#     def roofline_model(self, system: System):
#         device = system.device
#         interconnect = system.interconnect
#
#         qkv_latency = 3 * (
#                 self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
#         )
#         q_mul_k_latency = (
#                 self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
#         )
#         a_mul_v_latency = (
#                 self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
#         )
#         h_matmul0_latency = (
#                 self.H_matmul_o.roofline_model(device)
#                 + device.compute_module.overhead.matmul
#         )
#         h1_matmul1_latency = (
#                 self.H_matmul_1.roofline_model(device)
#                 + device.compute_module.overhead.matmul
#         )
#         h2_matmul2_latency = (
#                 self.H_matmul_2.roofline_model(device)
#                 + device.compute_module.overhead.matmul
#         )
#
#         matmul_total_latency = (
#                 qkv_latency
#                 + q_mul_k_latency
#                 + a_mul_v_latency
#                 + h_matmul0_latency
#                 + h1_matmul1_latency
#                 + h2_matmul2_latency
#         )
#
#         # normalization
#         softmax_latency = (
#                 self.A_softmax.roofline_model(device)
#                 + device.compute_module.overhead.softmax
#         )
#         layernorm_latency = (
#                 self.layer_norm_o.roofline_model(device)
#                 + device.compute_module.overhead.layernorm
#         )
#
#         normlization_total_latency = softmax_latency + layernorm_latency * 2
#
#         # gelu
#         gelu_latency = (
#                 self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
#         )
#
#         # allreduce
#         if self.device_count > 1:
#             allreduce_latency = self.allreduce_mha.simulate(interconnect)
#             allreduce_total_latency = allreduce_latency * 2
#         else:
#             allreduce_total_latency = 0
#             allreduce_total_latency = 0
#
#         # others
#
#         # print
#         print("Roofline breakdown:")
#         print(
#             f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
#         )
#         self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
#         print("total:")
#         print(
#             f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
#         )
#         self.roofline_latency = (
#                 matmul_total_latency
#                 + normlization_total_latency
#                 + gelu_latency
#                 + allreduce_total_latency
#         )
#         return self.roofline_latency
#
#     def compile_and_simulate(self, system: System, compile_mode: str):
#         device = system.device
#         interconnect = system.interconnect
#
#         # matmul
#         print("simulating qkv")
#         qkv_latency = 3 * (
#                 self.Q_proj.compile_and_simulate(device, compile_mode)
#                 + device.compute_module.overhead.matmul
#         )
#         print("simulating q_mul_k")
#         q_mul_k_latency = (
#                 self.Q_mul_K.compile_and_simulate(device, compile_mode)
#                 + device.compute_module.overhead.matmul
#         )
#         print("simulating a_mul_v")
#         a_mul_v_latency = (
#                 self.A_mul_V.compile_and_simulate(device, compile_mode)
#                 + device.compute_module.overhead.matmul
#         )
#         print("simulating h_matmul0")
#         h_matmul0_latency = (
#                 self.H_matmul_o.compile_and_simulate(device, compile_mode)
#                 + device.compute_module.overhead.matmul
#         )
#         print("simulating h1_matmul1")
#         h1_matmul1_latency = (
#                 self.H_matmul_1.compile_and_simulate(device, compile_mode)
#                 + device.compute_module.overhead.matmul
#         )
#         print("simulating h2_matmul2")
#         h2_matmul2_latency = (
#                 self.H_matmul_2.compile_and_simulate(device, compile_mode)
#                 + device.compute_module.overhead.matmul
#         )
#         print("finish matmul simulation")
#
#         matmul_total_latency = (
#                 qkv_latency
#                 + q_mul_k_latency
#                 + a_mul_v_latency
#                 + h_matmul0_latency
#                 + h1_matmul1_latency
#                 + h2_matmul2_latency
#         )
#
#         # normalization
#         softmax_latency = (
#                 self.A_softmax.compile_and_simulate(device, compile_mode)
#                 + device.compute_module.overhead.softmax
#         )
#         layernorm_latency = (
#                 self.layer_norm_o.compile_and_simulate(device, compile_mode)
#                 + device.compute_module.overhead.layernorm
#         )
#
#         normlization_total_latency = softmax_latency + layernorm_latency * 2
#
#         # gelu
#         gelu_latency = (
#                 self.H_gelu.compile_and_simulate(device, compile_mode)
#                 + device.compute_module.overhead.gelu
#         )
#
#         # allreduce
#         if self.device_count > 1:
#             allreduce_latency = self.allreduce_mha.simulate(interconnect)
#             allreduce_total_latency = allreduce_latency * 2
#         else:
#             allreduce_latency = 0
#             allreduce_total_latency = 0
#
#         # others
#
#         # print
#         # print("breakdown:")
#         # print(
#         #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
#         # )
#         # print("total:")
#         # print(
#         #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
#         # )
#         self.latency = (
#                 matmul_total_latency
#                 + normlization_total_latency
#                 + gelu_latency
#                 + allreduce_total_latency
#         )
#         self.simulate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
#         self.simulate_dict = {
#             'qkv': qkv_latency, 'q_mul_k': q_mul_k_latency, 'a_mul_v': a_mul_v_latency,
#             'h_matmul0': h_matmul0_latency, 'h1_matmul1': h1_matmul1_latency, 'h2_matmul2': h2_matmul2_latency,
#             'softmax': softmax_latency, 'layer_norm': layernorm_latency * 2, 'gelu': gelu_latency,
#             'allreduce': allreduce_latency * 2
#         }
#         return self.latency
#
#     def run_on_gpu(self):
#         # matmul
#         qkv_latency = (
#                           self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
#                       ) * 3
#         q_mul_k_latency = (
#             self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
#         )
#         a_mul_v_latency = (
#             self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
#         )
#         h_matmul0_latency = (
#             self.H_matmul_o.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
#         )
#         h1_matmul1_latency = (
#             self.H_matmul_1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
#         )
#         h2_matmul2_latency = (
#             self.H_matmul_2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
#         )
#
#         matmul_total_latency = (
#                 qkv_latency
#                 + q_mul_k_latency
#                 + a_mul_v_latency
#                 + h_matmul0_latency
#                 + h1_matmul1_latency
#                 + h2_matmul2_latency
#         )
#
#         # normalization
#         softmax_latency = (
#             self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
#         )
#         layernorm_latency = (
#                 self.layer_norm_o.run_on_gpu()
#                 - self.layer_norm_o.gpu_kernel_launch_overhead()
#         )
#
#         normlization_total_latency = softmax_latency + layernorm_latency * 2
#
#         # gelu
#         gelu_latency = (
#             self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
#         )
#
#         # allreduce
#         allreduce_total_latency = 0
#
#         # others
#
#         # print
#         print("breakdown:")
#         print(
#             f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
#         )
#         print("total:")
#         print(
#             f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
#         )
#         self.latency_on_gpu = (
#                 matmul_total_latency
#                 + normlization_total_latency
#                 + gelu_latency
#                 + allreduce_total_latency
#         )
#         return self.latency_on_gpu


class LlamaTransformerBlockTP(Operator):
    def __init__(self, model_args: ModelArgs, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.model_args = model_args

        # self.d_model = d_model
        # self.n_heads = n_heads

        self.d_model = self.model_args.dim
        self.n_heads = self.model_args.n_heads
        self.device_count = device_count

        # ffn_dim
        hidden_dim = model_args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if model_args.ffn_dim_multiplier is not None:
            hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = model_args.multiple_of * ((hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of)
        self.ffn_dim = hidden_dim

        # parameters per device
        d = self.d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.Wo = Tensor([d // device_count, d], data_type)

        self.W1 = Tensor([d, self.ffn_dim // device_count], data_type)
        self.W2 = Tensor([d, self.ffn_dim // device_count], data_type)
        self.W3 = Tensor([self.ffn_dim // device_count, d], data_type)

        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)

        # attention
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul_wo = Matmul(data_type)
        self.layer_norm_mha = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        # # feed-forward network
        self.H_matmul_w1 = Matmul(data_type)
        self.H_matmul_w2 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul_w3 = Matmul(data_type)
        self.layer_norm_block = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

        # TODO 添加 残差部分

    def __call__(self, x: Tensor, kv_cache_len: int = 0) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, s, d = x.shape
        assert d == self.d_model

        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        # KV cache
        K_cache = Tensor([b, h // dev_cnt, d_h, kv_cache_len], self.data_type)
        V_cache = Tensor([b, h // dev_cnt, kv_cache_len, d_h], self.data_type)  # 这个是reshape之后的

        # multi-head attention
        q = self.Q_proj(x, self.Wq)  # [b, s, d / dev_cnt]
        assert q.shape == [b, s, d // dev_cnt]
        k = self.K_proj(x, self.Wk)  # [b, s, d / dev_cnt]
        v = self.V_proj(x, self.Wv)  # [b, s, d / dev_cnt]

        q = self.Q_reshape(q, [b, s, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, s, h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, s, h // dev_cnt, d_h])

        q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h] 转置到可以拼接的 permute
        assert q_T.shape == [b, h // dev_cnt, s, d_h]
        k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, s]
        assert k_T.shape == [b, h // dev_cnt, d_h, s]
        v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
        assert v_T.shape == [b, h // dev_cnt, s, d_h]
        K_T = self.K_concat(K_cache, k_T, 3)  # [b, h / dev_cnt, d_h, s+1]
        assert K_T.shape == [b, h // dev_cnt, d_h, kv_cache_len + s]
        V_T = self.V_concat(V_cache, v_T, 2)  # [b, h / dev_cnt, s+1, d_h]
        assert V_T.shape == [b, h // dev_cnt, kv_cache_len + s, d_h]

        a = self.Q_mul_K(q_T, K_T)  # [b, h / dev_cnt, s, kv_cache_len + s]
        assert a.shape == [b, h // dev_cnt, s, kv_cache_len + s]
        a_prob = self.A_softmax(a)
        a_result = self.A_mul_V(a_prob, V_T)  # [b, h / dev_cnt, s, d_h]
        assert a_result.shape == [b, h // dev_cnt, s, d_h]
        a_result = self.H_transpose(a_result, [0, 2, 1, 3])  # [b, s, h / dev_cnt, d_h]
        assert a_result.shape == [b, s, h // dev_cnt, d_h]
        a_result = self.H_reshape(a_result, [b, s, d // dev_cnt])
        assert a_result.shape == [b, s, d // dev_cnt]
        a_output = self.H_matmul_wo(a_result, self.Wo)  # [b, s, d]
        assert a_output.shape == [b, s, d]
        x0 = self.layer_norm_mha(a_output)
        assert x0.shape == [b, s, d]
        if dev_cnt > 1:
            x0 = self.allreduce_mha(x0)

        # feed-forward network
        x1 = self.H_matmul_w1(x0, self.W1)  # [b, s, self.ffn_dim / dev_cnt]
        assert x1.shape == [b, s, self.ffn_dim // dev_cnt]
        x1 = self.H_gelu(x1)

        x2 = self.H_matmul_w2(x0, self.W2)  # [b, s, d]
        assert x2.shape == [b, s, self.ffn_dim // dev_cnt]

        # TODO silu and element mul

        output = self.H_matmul_w3(x2, self.W3)
        assert output.shape == [b, s, d]

        output = self.layer_norm_block(output)
        if dev_cnt > 1:
            output = self.allreduce_ffn(output)

        assert output.shape == [b, s, d]
        self.memory_requirement = (
                self.Wq.size * self.Wq.data_type.word_size
                + self.Wk.size * self.Wk.data_type.word_size
                + self.Wv.size * self.Wv.data_type.word_size
                + self.Wo.size * self.Wo.data_type.word_size
                + self.W1.size * self.W1.data_type.word_size
                + self.W2.size * self.W2.data_type.word_size
                + self.W3.size * self.W3.data_type.word_size
                + K_cache.size * K_cache.data_type.word_size
                + V_cache.size * V_cache.data_type.word_size
        )

        return output

    # def roofline_model(self, system: System):
    #     device = system.device
    #     interconnect = system.interconnect
    #
    #     qkv_latency = 3 * (
    #             self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
    #     )
    #     q_mul_k_latency = (
    #             self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
    #     )
    #     a_mul_v_latency = (
    #             self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
    #     )
    #     h_matmul0_latency = (
    #             self.H_matmul_wo.roofline_model(device)
    #             + device.compute_module.overhead.matmul
    #     )
    #     h1_matmul1_latency = (
    #             self.H_matmul_w1.roofline_model(device)
    #             + device.compute_module.overhead.matmul
    #     )
    #     h2_matmul2_latency = (
    #             self.H_matmul2.roofline_model(device)
    #             + device.compute_module.overhead.matmul
    #     )
    #
    #     matmul_total_latency = (
    #             qkv_latency
    #             + q_mul_k_latency
    #             + a_mul_v_latency
    #             + h_matmul0_latency
    #             + h1_matmul1_latency
    #             + h2_matmul2_latency
    #     )
    #
    #     # normalization
    #     softmax_latency = (
    #             self.A_softmax.roofline_model(device)
    #             + device.compute_module.overhead.softmax
    #     )
    #     layernorm_latency = (
    #             self.layer_norm_mha.roofline_model(device)
    #             + device.compute_module.overhead.layernorm
    #     )
    #
    #     normlization_total_latency = softmax_latency + layernorm_latency * 2
    #
    #     # gelu
    #     gelu_latency = (
    #             self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
    #     )
    #
    #     # allreduce
    #     if self.device_count > 1:
    #         allreduce_latency = self.allreduce_mha.simulate(interconnect)
    #         allreduce_total_latency = allreduce_latency * 2
    #     else:
    #         allreduce_latency = 0
    #         allreduce_total_latency = 0
    #
    #     # others
    #
    #     # print
    #     print("Roofline breakdown:")
    #     print(
    #         f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
    #     )
    #     print("total:")
    #     print(
    #         f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
    #     )
    #     self.roofline_latency = (
    #             matmul_total_latency
    #             + normlization_total_latency
    #             + gelu_latency
    #             + allreduce_total_latency
    #     )
    #     # print(f'memory requirement: {self.memory_requirement/1e9*96}GB')
    #     self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
    #     return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str):
        pcb = system.device
        interconnect = system.interconnect

        # matmul
        # print("simulating qkv")
        qkv_latency = 3 * (
                self.Q_proj.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.matmul
        )
        # print("simulating q_mul_k")
        q_mul_k_latency = (
                self.Q_mul_K.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.matmul
        )
        # print("simulating a_mul_v")
        a_mul_v_latency = (
                self.A_mul_V.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.matmul
        )
        # print("simulating h_matmul0")
        h_matmul_wo_latency = (
                self.H_matmul_wo.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.matmul
        )
        # print("simulating h1_matmul1")
        h_matmul_w1_latency = (
                self.H_matmul_w1.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.matmul
        )
        # print("simulating h2_matmul2")
        h_matmul_w2_latency = (
                self.H_matmul_w2.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.matmul
        )

        h_matmul_w3_latency = (
                self.H_matmul_w3.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.matmul
        )

        matmul_total_latency = (
                qkv_latency
                + q_mul_k_latency
                + a_mul_v_latency
                + h_matmul_wo_latency
                + h_matmul_w1_latency
                + h_matmul_w2_latency
                + h_matmul_w3_latency
        )

        # normalization
        softmax_latency = (
                self.A_softmax.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.softmax
        )
        layernorm_mha_latency = (
                self.layer_norm_mha.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.layernorm
        )
        layernorm_block_latency = (
                self.layer_norm_block.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.layernorm
        )

        layernorm_total_latency = layernorm_mha_latency + layernorm_block_latency

        normlization_total_latency = softmax_latency + layernorm_total_latency

        # gelu
        gelu_latency = (
                self.H_gelu.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_mha_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_ffn_latency = self.allreduce_ffn.simulate(interconnect)

            allreduce_total_latency = allreduce_mha_latency + allreduce_ffn_latency
        else:
            allreduce_mha_latency = 0
            allreduce_ffn_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )

        self.latency = (
                matmul_total_latency
                + normlization_total_latency
                + gelu_latency
                + allreduce_total_latency
        )

        self.simulate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul_wo_latency}, {h_matmul_w1_latency}, {h_matmul_w2_latency}, {softmax_latency}, {layernorm_mha_latency},{allreduce_ffn_latency}, {gelu_latency}, {allreduce_mha_latency},{allreduce_ffn_latency}"
        self.simulate_dict = {
            'qkv': qkv_latency, 'q_mul_k': q_mul_k_latency, 'a_mul_v': a_mul_v_latency,
            'h_matmul_wo': h_matmul_wo_latency, 'h_matmul_w1': h_matmul_w1_latency, 'h_matmul_w2': h_matmul_w2_latency,
            'h_matmul_w3': h_matmul_w3_latency,
            'softmax': softmax_latency, 'layer_norm_mha': layernorm_mha_latency,
            'layer_norm_block': layernorm_block_latency, 'gelu': gelu_latency,
            'allreduce_mha': allreduce_mha_latency, "allreduce_ffn": allreduce_ffn_latency
        }
        return self.latency

    # def run_on_gpu(self):
    #     # matmul
    #     qkv_latency = (
    #                       self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
    #                   ) * 3
    #     q_mul_k_latency = (
    #         self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
    #     )
    #     a_mul_v_latency = (
    #         self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
    #     )
    #     h_matmul0_latency = (
    #         self.H_matmul_wo.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
    #     )
    #     h1_matmul1_latency = (
    #         self.H_matmul_w1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
    #     )
    #     h2_matmul2_latency = (
    #         self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
    #     )
    #
    #     matmul_total_latency = (
    #             qkv_latency
    #             + q_mul_k_latency
    #             + a_mul_v_latency
    #             + h_matmul0_latency
    #             + h1_matmul1_latency
    #             + h2_matmul2_latency
    #     )
    #
    #     # normalization
    #     softmax_latency = (
    #         self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
    #     )
    #     layernorm_latency = (
    #             self.layer_norm_mha.run_on_gpu()
    #             - self.layer_norm_mha.gpu_kernel_launch_overhead()
    #     )
    #
    #     normlization_total_latency = softmax_latency + layernorm_latency * 2
    #
    #     # gelu
    #     gelu_latency = (
    #         self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
    #     )
    #     # gelu_latency = max(gelu_latency, 1e-7)
    #
    #     # allreduce
    #     allreduce_total_latency = 0
    #
    #     # others
    #
    #     # print
    #     print("breakdown:")
    #     print(
    #         f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
    #     )
    #     print("total:")
    #     print(
    #         f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
    #     )
    #     self.latency_on_gpu = (
    #             matmul_total_latency
    #             + normlization_total_latency
    #             + gelu_latency
    #             + allreduce_total_latency
    #     )
    #     return self.latency_on_gpu


class LLMInitComputationTP:
    def __init__(
            self,
            d_model,
            n_heads,
            n_layers,
            device_count,
    ) -> None:
        pass
