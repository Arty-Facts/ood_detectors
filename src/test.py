from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

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
            print(f"  GPU Utilization: {nvmlDeviceGetUtilizationRates(handle).gpu} %")
            print(f"  Memory Utilization: {nvmlDeviceGetUtilizationRates(handle).memory} %") 

    except Exception as e:
        print("Error accessing NVML:", e)

get_nvml_info()