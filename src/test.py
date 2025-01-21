from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
import time
def get_nvml_info():
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        print(f"Found {device_count} GPU(s):")
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            memory = nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU {i}: {name}")
            # in
            print(f"  Total Memory: {memory.total * 1e-9:.2f} GB")
            print(f"  Free Memory: {memory.free * 1e-9:.2f} GB")
            print(f"  Used Memory: {memory.used * 1e-9:.2f} GB")


            # utilization 
            gpu_util = 0.0
            mem_util = 0.0
            for i in range(10):
                gpu_util += nvmlDeviceGetUtilizationRates(handle).gpu
                mem_util += nvmlDeviceGetUtilizationRates(handle).memory
                time.sleep(0.1)

            gpu_util /= 10
            mem_util /= 10

            print(f"  GPU Utilization: {gpu_util}%")
            print(f"  Memory Utilization: {mem_util}%")

    except Exception as e:
        print("Error accessing NVML:", e)

get_nvml_info()