from util.func import *
from util.workflow import test_fam



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("load_mesh_path", type=str)
    parser.add_argument("load_ckpt_path", type=str)
    parser.add_argument("load_check_map_path", type=str)
    parser.add_argument("export_folder", type=str)
    parser.add_argument("input_format", type=str, choices=["mesh_verts", "sampled_points"])
    parser.add_argument("--N_poisson_approx", type=int, default=100000)
    args = parser.parse_args()
    
    suffix = f"tested_on_{args.input_format}"
    save_uv_image_path = os.path.join(args.export_folder, f"UV_{suffix}.png")
    save_edge_points_path = os.path.join(args.export_folder, f"edge_points_{suffix}.ply")
    save_textured_points_path = os.path.join(args.export_folder, f"textured_points_{suffix}.ply")
    
    vtx_pos, vtx_nor, tri_vid = load_mesh_with_normalization(args.load_mesh_path, True, True)
    # vtx_pos: (num_verts, 3)
    # vtx_nor: (num_verts, 3)
    # tri_vid: (num_faces, 3)
    num_verts = vtx_pos.shape[0]
    num_faces = tri_vid.shape[0]
    print(f"{num_verts=}, {num_faces=}")
    
    if args.input_format == "mesh_verts":
        f_pos_c = vtx_pos[tri_vid].mean(axis=1, keepdims=True)
        f_nor_c = vtx_nor[tri_vid].mean(axis=1, keepdims=True)
        f_pos_i = (vtx_pos[tri_vid] + f_pos_c) / 2
        f_nor_i = (vtx_nor[tri_vid] + f_nor_c) / 2

        vtx_pos = np.concatenate([vtx_pos, f_pos_c.reshape(-1, 3), f_pos_i.reshape(-1, 3)], axis=0)
        vtx_nor = np.concatenate([vtx_nor, f_nor_c.reshape(-1, 3), f_nor_i.reshape(-1, 3)], axis=0)

        vtx_uv_image_list, vtx_uv, vtx_edge, vtx_checker_colors = test_fam(vtx_pos, vtx_nor, args.load_ckpt_path, args.load_check_map_path, args.export_folder)
        one_row_export_image_list(vtx_uv_image_list, ["Q_hat", "Q_hat_cycle", "Q"], 4.0, 8.0, save_uv_image_path)
        if vtx_edge is not None:
            save_ply_point_cloud(save_edge_points_path, vtx_edge)
        save_ply_point_cloud(save_textured_points_path, vtx_pos, vtx_checker_colors, vtx_nor)

        # normalize uv
        vtx_uv = (vtx_uv - vtx_uv.min(0)) / (vtx_uv.max(0) - vtx_uv.min(0) + 1e-5)

        eps = 1e-2

        face_uv_c = vtx_uv[num_verts:num_verts+num_faces]
        face_uv_i = vtx_uv[num_verts+num_faces:]
        
        faces_uvs = []
        addon_uvs = []
        for i in range(num_faces):
            face = tri_vid[i]

            if np.linalg.norm(vtx_uv[face].mean(0) - face_uv_c[i]) > eps:
                v_0, v_1, v_2 = face
                
                if np.linalg.norm((face_uv_c[i] + vtx_uv[face[0]]) / 2 - face_uv_i[3*i+0] ) > eps:
                    v_0 = len(addon_uvs) + num_verts
                    if np.linalg.norm(face_uv_c[i] - face_uv_i[3*i+0]) < eps:
                        addon_uvs.append(face_uv_i[3*i+0])
                    else:
                        addon_uvs.append(face_uv_c[i])
                if np.linalg.norm((face_uv_c[i] + vtx_uv[face[1]]) / 2 - face_uv_i[3*i+1] ) > eps:
                    v_1 = len(addon_uvs) + num_verts
                    if np.linalg.norm(face_uv_c[i] - face_uv_i[3*i+1]) < eps:
                        addon_uvs.append(face_uv_i[3*i+1])
                    else:
                        addon_uvs.append(face_uv_c[i])
                if np.linalg.norm((face_uv_c[i] + vtx_uv[face[2]]) / 2 - face_uv_i[3*i+2] ) > eps:
                    v_2 = len(addon_uvs) + num_verts
                    if np.linalg.norm(face_uv_c[i] - face_uv_i[3*i+2]) < eps:
                        addon_uvs.append(face_uv_i[3*i+2])
                    else:
                        addon_uvs.append(face_uv_c[i])

                faces_uvs.append(np.array([v_0, v_1, v_2]))
            else:
                faces_uvs.append(face)
        
        out_uvs = np.concatenate([vtx_uv[:num_verts], np.array(addon_uvs)], axis=0)
        faces_uvs = np.array(faces_uvs)

        # write obj file
        with open(args.export_folder + "/UV.obj", "w") as f:
            
            for i in range(num_verts):
                f.write(f"v {vtx_pos[i, 0]} {vtx_pos[i, 1]} {vtx_pos[i, 2]}\n")
            for i in range(len(out_uvs)):
                f.write(f"vn {vtx_nor[i, 0]} {vtx_nor[i, 1]} {vtx_nor[i, 2]}\n")
            for i in range(len(out_uvs)):
                f.write(f"vt {out_uvs[i, 0]} {out_uvs[i, 1]}\n")
            for i in range(len(faces_uvs)):
                f.write(f"f {tri_vid[i, 0]+1}/{faces_uvs[i, 0]+1}/{tri_vid[i, 0]+1} {tri_vid[i, 1]+1}/{faces_uvs[i, 1]+1}/{tri_vid[i, 1]+1} {tri_vid[i, 2]+1}/{faces_uvs[i, 2]+1}/{tri_vid[i, 2]+1}\n")
        
    if args.input_format == "sampled_points":
        poisson_points, poisson_normals = sample_points_from_mesh_approx(vtx_pos, tri_vid, args.N_poisson_approx, vtx_nor)
        poisson_normals = normalize_normals(poisson_normals) # After sampling, the normal values may slightly overflow [-1, +1].
        N_poisson = poisson_points.shape[0]
        print(f"actual number of sampled points: {N_poisson}")
        pts_uv_image_list, pts_uv, pts_edge, pts_checker_colors = test_fam(poisson_points, poisson_normals, args.load_ckpt_path, args.load_check_map_path, args.export_folder)
        one_row_export_image_list(pts_uv_image_list, ["Q_hat", "Q_hat_cycle", "Q"], 4.0, 8.0, save_uv_image_path)
        if pts_edge is not None:
            save_ply_point_cloud(save_edge_points_path, pts_edge)
        save_ply_point_cloud(save_textured_points_path, poisson_points, pts_checker_colors, poisson_normals)


if __name__ == '__main__':
    main()
    print("testing finished.")
