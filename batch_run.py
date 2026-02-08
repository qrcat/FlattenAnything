import os
import json

if __name__ == '__main__':
    dir = 'example/input_model/uv_islands'

    for file in os.listdir(dir):
        if file.endswith('.obj'):
            path = os.path.join(dir, file)
            basename = os.path.basename(file).split('.')[0]
            # os.system(f'python train_mesh.py "{path}" "./exported" 10000 10000')
            os.system(f'python test_mesh.py "{path}" "./exported/{basename}/fam.pth" "./example/checker_map/20x20.png" "./exported/{basename}" "mesh_verts"')

            with open(f"./exported/{basename}/meta.json", 'w') as f:
                json.dump({
                    "path": "./exported/{basename}/UV.obj",
                    "gt": path,
                }, f)
