import numpy as np
import tensorflow as tf
import torch
import keras
from keras import ops

def run_tensor_tasks(data: np.ndarray, k: int = 5, p: int = 4):
    """
    Задания 38-45: Работа с тензорами TensorFlow, PyTorch и Keras.
    data: массив формы (n, m)
    """
    n, m = data.shape

    # Задание 38. Тензор TensorFlow
    X_tf = tf.constant(data, dtype=tf.float32)

    # Задание 39. Тензор PyTorch
    X_pt = torch.tensor(data, dtype=torch.float32)

    # Задание 40. Случайные тензоры
    # A — целочисленный (k, n), W — нормальный (m, p), B — равномерный (k, p)
    A_tf = tf.random.uniform((k, n), minval=0, maxval=10, dtype=tf.int32)
    W_tf = tf.random.normal((m, p), dtype=tf.float32)
    B_tf = tf.random.uniform((k, p), dtype=tf.float32)

    # Задание 41. Операция A @ X @ W + B в TensorFlow
    AX_tf = tf.matmul(tf.cast(A_tf, tf.float32), X_tf)
    AXW_tf = tf.matmul(AX_tf, W_tf)
    res_tf = AXW_tf + B_tf

    # Задание 42. Та же операция A @ X @ W + B в PyTorch
    A_pt = torch.randint(0, 10, (k, n), dtype=torch.float32)
    W_pt = torch.randn(m, p, dtype=torch.float32)
    B_pt = torch.rand(k, p, dtype=torch.float32)
    res_pt = A_pt @ X_pt @ W_pt + B_pt

    # Задание 43. Матрицы T, P, Q
    T = np.random.rand(3, 10)
    P = np.random.rand(3, 10)
    Q = np.random.rand(3, 10)

    # Задание 44. Операция в Keras Core Ops: |sin(T) - exp(P) * sqrt(Q)|
    T_k = ops.convert_to_tensor(T)
    P_k = ops.convert_to_tensor(P)
    Q_k = ops.convert_to_tensor(Q)
    V_k = ops.abs(ops.sin(T_k) - ops.exp(P_k) * ops.sqrt(Q_k))

    # Задание 45. Та же операция в PyTorch
    T_t = torch.tensor(T)
    P_t = torch.tensor(P)
    Q_t = torch.tensor(Q)
    V_t = torch.abs(torch.sin(T_t) - torch.exp(P_t) * torch.sqrt(Q_t))

    return {
        "res_tf_shape": res_tf.shape,
        "res_pt_shape": res_pt.shape,
        "keras_pytorch_match": np.allclose(ops.convert_to_numpy(V_k), V_t.numpy(), atol=1e-6)
    }