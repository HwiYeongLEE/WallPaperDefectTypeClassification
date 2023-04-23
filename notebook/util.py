from pynvml import *
import matplotlib.pyplot as plt


def show_img(img, channel_first=True, figsize=(5, 5)):
    img = img.T if channel_first else img
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.axis(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img)
    plt.imshow(img)


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
