import time, random
import multiprocessing as mp

class dummy:
    @staticmethod
    def is_alive():
        return False
    @staticmethod
    def close():
        pass
    @staticmethod
    def terminate():
        pass
    @staticmethod
    def join():
        pass
    @staticmethod
    def kill():
        pass

class Worker:
    def __init__(self, id, node):
        self.id = id
        self.node = node
        self.process = dummy
        self.start_time = time.perf_counter()


def parallelize(func, jobs, gpu_nodes, verbose=True, timeout=3600):
    """
    Parallelize the execution of a function on multiple GPUs.

    Args:
        func (function): The function to be executed in parallel.
        jobs (list): A list of jobs to be executed by the function.
        gpu_nodes (list): A list of GPU nodes to be used for execution.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        timeout (int, optional): Timeout in seconds for each job. Defaults to 3600.

    Returns:
        None
    """
    if verbose:
        print(f'Launching {len(jobs)} jobs on {len(set(gpu_nodes))} GPUs. {len(gpu_nodes)//len(set(gpu_nodes))} jobs per GPU in parallel..')
    workers = [Worker(id, node) for id, node in enumerate(gpu_nodes)]
    while len(jobs) > 0:
        random.shuffle(workers)
        for worker in workers:
            if time.perf_counter() - worker.start_time > timeout:
                if verbose:
                    print(f'Job on cuda:{worker.node} in slot {worker.id} timed out. Killing it...')
                worker.process.terminate()
            if not worker.process.is_alive():
                worker.process.kill()
                if verbose:
                    print(f'Launching job on cuda:{worker.node} in slot {worker.id}. {len(jobs)} jobs to left...')
                if len(jobs) == 0:
                    break
                args = list(jobs.pop())
                args.append(f'cuda:{worker.node}')
                p = mp.Process(target=func, args=args)
                p.start()
                worker.process = p
                worker.start_time = time.perf_counter()
                time.sleep(1)
        time.sleep(1)
    for worker in workers:
        worker.process.join()
    if verbose:
        print('Done!')