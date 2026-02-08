import os
from collections import defaultdict, deque

def load_obj(path):
    V, VT, VN = [], [], []
    faces = []  # [( (v,vt,vn), (v,vt,vn), (v,vt,vn) )]

    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                V.append(tuple(map(float, line.split()[1:])))
            elif line.startswith("vt "):
                VT.append(tuple(map(float, line.split()[1:])))
            elif line.startswith("vn "):
                VN.append(tuple(map(float, line.split()[1:])))
            elif line.startswith("f "):
                face = []
                for tok in line.split()[1:]:
                    vals = tok.split("/")
                    v  = int(vals[0]) - 1
                    vt = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
                    vn = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else None
                    face.append((v, vt, vn))
                faces.append(tuple(face))

    return V, VT, VN, faces


def build_uv_face_graph(faces):
    """
    Face adjacency graph based on shared UV edges
    """
    edge2faces = defaultdict(list)

    for fi, face in enumerate(faces):
        for i in range(3):
            a = face[i][1]
            b = face[(i + 1) % 3][1]
            if a is None or b is None:
                continue
            edge = tuple(sorted((a, b)))
            edge2faces[edge].append(fi)

    graph = defaultdict(set)
    for fs in edge2faces.values():
        if len(fs) > 1:
            for i in fs:
                for j in fs:
                    if i != j:
                        graph[i].add(j)

    return graph


def connected_components(graph, n_faces):
    visited = [False] * n_faces
    components = []

    for i in range(n_faces):
        if visited[i]:
            continue
        queue = deque([i])
        visited[i] = True
        comp = []

        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        components.append(comp)

    return components


def export_obj(path, V, VT, VN, faces, face_ids):
    v_map, vt_map, vn_map = {}, {}, {}
    new_V, new_VT, new_VN = [], [], []
    new_faces = []

    def remap(idx, mapping, storage):
        if idx not in mapping:
            mapping[idx] = len(storage)
            storage.append(idx)
        return mapping[idx]

    for fi in face_ids:
        face = faces[fi]
        new_face = []
        for v, vt, vn in face:
            nv  = remap(v,  v_map,  new_V)
            nvt = remap(vt, vt_map, new_VT) if vt is not None else None
            nvn = remap(vn, vn_map, new_VN) if vn is not None else None
            new_face.append((nv, nvt, nvn))
        new_faces.append(new_face)

    with open(path, "w") as f:
        for i in new_V:
            f.write(f"v {V[i][0]} {V[i][1]} {V[i][2]}\n")
        for i in new_VT:
            f.write(f"vt {VT[i][0]} {VT[i][1]}\n")
        for i in new_VN:
            f.write(f"vn {VN[i][0]} {VN[i][1]} {VN[i][2]}\n")

        for face in new_faces:
            line = "f"
            for v, vt, vn in face:
                v += 1
                vt = vt + 1 if vt is not None else ""
                vn = vn + 1 if vn is not None else ""
                if vt != "" or vn != "":
                    line += f" {v}/{vt}/{vn}"
                else:
                    line += f" {v}"
            f.write(line + "\n")


def split_obj_by_uv(obj_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    V, VT, VN, faces = load_obj(obj_path)
    graph = build_uv_face_graph(faces)
    components = connected_components(graph, len(faces))

    print(f"Found {len(components)} UV islands")

    for i, comp in enumerate(components):
        out_path = os.path.join(out_dir, f"mesh_uv_{i}.obj")
        export_obj(out_path, V, VT, VN, faces, comp)


if __name__ == "__main__":
    split_obj_by_uv("example/input_model/smplx_uv.obj", "example/input_model/uv_islands")
