import numpy as np
import torch
import torch.fft as fft


def Fourier_filter_1d_func_compose(x_ori, x_back, cond_para):
    """Fuse two token streams in the frequency domain for DEIF."""
    x_ori = x_ori.permute(0, 2, 1).to(dtype=torch.float32)
    x_back = x_back.permute(0, 2, 1).to(dtype=torch.float32)
    _, _, length = x_ori.shape

    x_ori_freq = fft.fftshift(fft.fft(x_ori, dim=-1), dim=-1)
    x_back_freq = fft.fftshift(fft.fft(x_back, dim=-1), dim=-1)

    weights = function_interpolation(length, cond_para)
    weights = torch.from_numpy(weights).to(x_ori_freq.device, dtype=torch.float32)
    weights = weights.reshape(1, 1, length)

    fused_freq = weights * x_ori_freq + (1 - weights) * x_back_freq
    fused = fft.ifft(fft.ifftshift(fused_freq, dim=-1), dim=-1).real
    return fused.to(dtype=torch.float16).permute(0, 2, 1)


def function_interpolation(length, cond_para):

    def gaussian_func(x, sigma=0.3, k=None):
        x_norm = (x - length / 2) / (length / 2)
        return np.exp(-(x_norm**2) / (2 * sigma**2))

    def gaussian_controlled(x, k=6.0, sigma=250):
        x_norm = np.abs(x - length / 2)
        return np.exp(-((x_norm / sigma) ** k))

    func_map = {
        "gaussian_func": gaussian_func,
        "gaussian_controlled": gaussian_controlled,
    }
    func_name = cond_para["func"]
    if func_name == "linear_inter":
        return np.full(length, cond_para["condweight"], dtype=np.float32)
    if func_name not in func_map:
        raise ValueError(f"Unknown frequency fusion function: {func_name}")

    x_values = np.arange(0, length)
    y_values = func_map[func_name](x_values, k=cond_para["k"], sigma=cond_para["sigma"])
    y_values = 1 - y_values if cond_para["reverse"] else y_values
    return y_values.astype(np.float32)
