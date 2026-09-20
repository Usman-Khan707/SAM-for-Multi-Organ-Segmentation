import numpy as np
import trimesh
from skimage import measure
from tqdm import tqdm


def visualization_3d_mesh(data_path, output_path):
    data = np.load(data_path)
    data[data == 0] = -1
    mesh_list = []
    for label in tqdm(range(1, 13 + 1)):
        color = np.random.randint(0, 255, size=(3,)) * 0.5
        data_copy = data.copy()
        data_copy[data_copy != label] = -1
        data_copy[data_copy == label] = 1
        if np.where(data_copy == 1)[0].shape[0] == 0:
            continue
        vertices, faces, normals, _ = measure.marching_cubes(volume=data_copy, level=0, spacing=(0.2, 0.2, 1.0))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=normals, process=False,
        face_colors=color
        )
        mesh.fix_normals()
        mesh_list.append(mesh)
    mesh = trimesh.util.concatenate(mesh_list)
    # mesh.show()
    mesh.export(output_path)


if __name__ == '__main__':
    data_path = './results.npy'
    output_path = './mesh.obj'
    visualization_3d_mesh(data_path, output_path)