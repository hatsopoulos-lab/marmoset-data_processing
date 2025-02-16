import torch

print('CUDA available', torch.cuda.is_available(),flush=True)
print('CUDA version', torch.version.cuda, flush=True)
print('cuDNN enabled', torch.backends.cudnn.enabled, flush=True)
print('cuDNN version', torch.backends.cudnn.version(), flush=True)

n_cuda_devices = torch.cuda.device_count()
for i in range(n_cuda_devices):
  print(f'Device {i} name:', torch.cuda.get_device_name(i), flush=True)
