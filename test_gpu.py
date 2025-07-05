import cupy as cp
print("Num GPUs:", cp.cuda.runtime.getDeviceCount())
x = cp.arange(10)
print(x)