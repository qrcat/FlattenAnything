import os
import json


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

    for path in data:
        basename = os.path.basename(path).split('.')[0]
        os.system(f'python train_mesh.py "{path}" "./exported" 10000 1000')
        os.system(f'python test_mesh.py "{path}" "./exported/{basename}/fam.pth" "./example/checker_map/20x20.png" "./exported/{basename}" "mesh_verts"')

        with open(f"./exported/{basename}/meta.json", 'w') as f:
            json.dump({
                "path": "./exported/{basename}/UV.obj",
                "gt": path,
            }, f)
