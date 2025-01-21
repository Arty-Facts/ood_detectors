from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo

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
            print(f"  Total Memory: {memory.total / 1024**2:.2f} MB")
            print(f"  Used Memory: {memory.used / 1024**2:.2f} MB")
            print(f"  Free Memory: {memory.free / 1024**2:.2f} MB")
    except Exception as e:
        print("Error accessing NVML:", e)

get_nvml_info()