import numpy as np
from scipy.spatial import cKDTree

def rotationZ(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

def remap(target, target_prox):
    mapping = np.arange(target_prox.shape[0])
    # build KD tree for quicker search
    target_KDtree = cKDTree(target)
    for i in range(target_prox.shape[0]):
        _, mapping[i] = target_KDtree.query(target_prox[i])

    return mapping

def ICP(source, target, iters = 8, z_angle = 0.0, p = None):
    mean_source = np.mean(source, axis=0)
    delta_source = source - mean_source
    mean_target = np.mean(target, axis=0)
    delta_target = target - mean_target
    # try to initialize different rotation matrix
    rotation_matrix = rotationZ(z_angle)
    if (p == None):
        p = np.zeros(3)

    for iter in range(iters):
        # map to closest point
        target_prox = (rotation_matrix.T @ (source - p).T).T
        mapping = remap(target, target_prox)

        # calculate dist of point cloud
        dist = 0
        for i in range(source.shape[0]):
            dist = dist + np.linalg.norm(target[mapping[i]] - target_prox[i])
        print(f"iter: {iter}, dist = {dist}")

        # iterate next rotation matrix and p
        Q = np.zeros((3,3))
        for j in range(source.shape[0]):
            Q = Q + delta_source[j].reshape((3, 1)) @ delta_target[mapping[j]].reshape((1, 3))
        U, _, V = np.linalg.svd(Q)
        rotation_matrix = U @ V
        if np.linalg.det(rotation_matrix) < 0.0:
            V[-1, :] *= -1
            rotation_matrix = U @ V
        p = mean_source - (rotation_matrix @ mean_target)

    pose = np.eye(4)
    pose[:3,:3] = rotation_matrix.T
    pose[:3, -1] = -(rotation_matrix.T @ p)
    print(f"Pose: \n{pose}")
    return pose