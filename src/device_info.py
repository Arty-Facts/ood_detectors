from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
import time

class Memory():
    def __init__(self, total, free, used):
        self.total = total
        self.free = free
        self.used = used

    def __repr__(self):
        return f"Memory(total={self.total:.2f}GB, free={self.free:.2f}GB, used={self.used:.2f}GB)" 

class Util():
    def __init__(self, gpu, memory):
        self.gpu = gpu
        self.memory = memory
        
    def __repr__(self):
        return f"Util(gpu={self.gpu:.2f}%, memory={self.memory:.2f}%)"

class GPU():
    def __init__(self, name, mem_info, util_info):
        self.name = name
        self.mem = mem_info
        self.util = util_info

    def __repr__(self):
        return f"GPU(name={self.name}, util={self.util} mem={self.mem})"

class Device():
    def __init__(self):
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()

        self.gpu_info = []

        for i in range(self.device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)

            gpu_util = 0.0
            mem_util = 0.0
            mem_free = 0.0
            mem_total = 0.0
            mem_used = 0.0
            for i in range(10):
                memory = nvmlDeviceGetMemoryInfo(handle)
                gpu_util += nvmlDeviceGetUtilizationRates(handle).gpu
                mem_util += nvmlDeviceGetUtilizationRates(handle).memory
                mem_free += memory.free
                mem_total += memory.total
                mem_used += memory.used
                time.sleep(0.1)

            gpu_util /= 10
            mem_util /= 10
            mem_free /= 10
            mem_total /= 10
            mem_used /= 10

            to_gb = 1024**3
            mem_free /= to_gb
            mem_total /= to_gb
            mem_used /= to_gb

            self.gpu_info.append(GPU(
                name, 
                Memory(mem_total, mem_free, mem_used),
                Util(gpu_util, mem_util))
            )
        
        nvmlShutdown()

    def __len__(self):
        return self.device_count
    
    def __getitem__(self, i):
        return self.gpu_info[i]
    
    def __repr__(self):
        return f"Device({self.gpu_info})"
    
if __name__ == "__main__":
    device = Device()
    for i in range(len(device)):
        print(device[i])