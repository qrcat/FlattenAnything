from pathlib import Path
from tqdm import tqdm

import os
import time
import json
import argparse
import multiprocessing


def worker(queue, count, device) -> None:
    while True:
        path = queue.get()
        if path is None:
            break

        basename = os.path.basename(path).split('.')[0]
        os.system(f'CUDA_VISIBLE_DEVICES={device} python train_mesh.py "{path}" "./exported" 10000 10000')
        os.system(f'CUDA_VISIBLE_DEVICES={device} python test_mesh.py "{path}" "./exported/{basename}/fam.pth" "./example/checker_map/20x20.png" "./exported/{basename}" "mesh_verts"')

        with open(f"./exported/{basename}/meta.json", 'w') as f:
            json.dump({
                "path": f"./exported/{basename}/UV.obj",
                "gt": path,
            }, f)
        
        with count.get_lock():
            count.value += 1
        
        queue.task_done()


def fetch_data(path):
    if os.path.isdir(path):
        data = []
        for file in os.listdir(path):
            if file.endswith('.obj'):
                data.append(os.path.join(path, file))
        return data
    
    suffix = path.split('.')[-1]
    if suffix == 'obj':
        return [path]
    elif suffix == 'json':
        with open(path, 'r') as f:
            data = json.load(f)
        return [item['path'] for item in data if item['num_faces'] >= 4]
    else:
        raise ValueError(f"Invalid path: {path}")

if __name__ == '__main__':
    
    # data = fetch_data('example/input_model/uv_islands')
    data = fetch_data('test.json')

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    processes = []

    workers = 32

    # use parallel processing
    for worker_i in range(workers):
        process = multiprocessing.Process(
            target=worker,
            args=(queue, count, worker_i % 8)
        )
        process.daemon = True
        process.start()
        processes.append(process)

    for path in data:
        queue.put(path)

    start_time = time.time()
    queue.join()
    end_time = time.time()
    print(f"All tasks completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {count.value} items")

    for worker_i in range(workers):
        queue.put(None)

    for p in processes:
        p.join()
    print("All worker processes finished")

