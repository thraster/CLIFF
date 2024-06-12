'''
可视化demo的运行结果
SLP subject 102 上的运行结果

'''
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial import procrustes


def visualize_point_cloud(vertices, color=[1, 0, 0]):
    """
    使用 Open3D 可视化单个点云
    :param vertices: numpy 数组，形状为 (6890, 3)
    :param color: list，点云颜色
    :return: open3d.geometry.PointCloud 对象
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    point_cloud.paint_uniform_color(color)
    return point_cloud

def visualize_mesh(vertices, faces, color=[1, 0, 0], alpha=1.0):
    """
    使用 Open3D 将点云重建为网格并可视化
    :param vertices: numpy 数组，形状为 (6890, 3)
    :param faces: numpy 数组，形状为 (N, 3)
    :param color: list，网格颜色
    :param alpha: float，网格透明度
    :return: open3d.geometry.TriangleMesh 对象
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    # 设置颜色和透明度
    rgba_color = np.array(color + [alpha])  # 将颜色和透明度结合
    vertex_colors = np.tile(rgba_color, (vertices.shape[0], 1))  # 为每个顶点设置颜色和透明度
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[:, :3])  # 设置顶点颜色
    
    return mesh

def visualize_wireframe(vertices, faces, color=[1, 0, 0]):
    """
    使用 Open3D 将点云重建为线框网格并可视化
    :param vertices: numpy 数组，形状为 (6890, 3)
    :param faces: numpy 数组，形状为 (N, 3)
    :param color: list，线框颜色
    :return: open3d.geometry.LineSet 对象
    """
    lines = []
    for face in faces:
        lines.append([face[0], face[1]])
        lines.append([face[1], face[2]])
        lines.append([face[2], face[0]])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

# -------------------------- metrics --------------------------

def procrustes_3d(X, Y):
    """
    Perform Procrustes alignment of 3D points.
    
    Args:
    - X (np.array): Source points, shape (N, 3)
    - Y (np.array): Target points, shape (N, 3)
    
    Returns:
    - Z (np.array): Aligned source points, shape (N, 3)
    """
    # Translate X and Y to their centroids
    X_centroid = X.mean(axis=0)
    Y_centroid = Y.mean(axis=0)
    X = X - X_centroid
    Y = Y - Y_centroid
    
    # Compute the covariance matrix
    covariance_matrix = np.dot(Y.T, X)
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(covariance_matrix)
    
    # Compute the rotation matrix
    R = np.dot(U, Vt)
    
    # Apply the rotation to X
    Z = np.dot(X, R)
    
    # Scale the points
    scale = np.trace(np.dot(Z.T, Y)) / np.trace(np.dot(Z.T, Z))
    Z *= scale
    
    # Translate back to the target's centroid
    Z += Y_centroid
    
    return Z

def batch_compute_similarity_transform_torch(S1, S2):
    '''
    将S1与S2对齐，返回调整后的S1和转移矩阵
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat, (scale, R, t)


def calculate_single_v2v(pred_vertices, gt_vertices):
    """
    Calculate V2V between predicted and ground truth vertices for a single sample.
    
    Args:
    - pred_vertices (np.array): Predicted vertex positions, shape (V, 3)
    - gt_vertices (np.array): Ground truth vertex positions, shape (V, 3)
    
    Returns:
    - v2v (float): V2V error for the single sample
    """
    # Calculate V2V
    v2v = np.sqrt(np.sum((pred_vertices - gt_vertices) ** 2, axis=1)).mean()
    return v2v

def calculate_batch_v2v(pred_batch, gt_batch):
    """
    Calculate V2V for a batch of predicted and ground truth vertices.
    
    Args:
    - pred_batch (np.array): Predicted vertex positions, shape (N, V, 3)
    - gt_batch (np.array): Ground truth vertex positions, shape (N, V, 3)
    
    Returns:
    - v2v_list (list): List of V2V errors for each sample in the batch
    """
    batch_size = pred_batch.shape[0]
    v2v_list = []


    for i in range(batch_size):
        v2v = calculate_single_v2v(pred_batch[i], gt_batch[i])
        v2v_list.append(v2v)
    v2v_array = np.array(v2v_list)

    v2v = {
        'mean': np.mean(v2v_array),
        'std': np.std(v2v_array),
        'min': np.min(v2v_array),
        'max': np.max(v2v_array),
        'median': np.median(v2v_array)
    }
    
    return v2v


def calculate_single_mpjpe(pred_joints, gt_joints):
    """
    Calculate MPJPE between predicted and ground truth joints for a single sample.
    
    Args:
    - pred_joints (np.array): Predicted joint positions, shape (K, 3)
    - gt_joints (np.array): Ground truth joint positions, shape (K, 3)
    
    Returns:
    - mpjpe (float): MPJPE error for the single sample
    """
    # Calculate MPJPE
    mpjpe = np.sqrt(np.sum((pred_joints - gt_joints) ** 2, axis=1)).mean()
    return mpjpe

def calculate_batch_mpjpe(pred_batch, gt_batch):
    """
    Calculate MPJPE for a batch of predicted and ground truth joints.
    
    Args:
    - pred_batch (np.array): Predicted joint positions, shape (N, K, 3)
    - gt_batch (np.array): Ground truth joint positions, shape (N, K, 3)
    
    Returns:
    - mpjpe_list (list): List of MPJPE errors for each sample in the batch
    """
    batch_size = pred_batch.shape[0]
    mpjpe_list = []

    for i in range(batch_size):
        mpjpe = calculate_single_mpjpe(pred_batch[i], gt_batch[i])
        mpjpe_list.append(mpjpe)
    
    mpjpe_array = np.array(mpjpe_list)

    mpjpe = {
        'mean': np.mean(mpjpe_array),
        'std': np.std(mpjpe_array),
        'min': np.min(mpjpe_array),
        'max': np.max(mpjpe_array),
        'median': np.median(mpjpe_array)
    }

    return mpjpe

# -------------------------------- viz ---------------------------------------

def compare_vertices(pred, gt, smpl_model):
    """
    对比预测和真实的 vertices，使用 Open3D 可视化
    :param pred_vertices: numpy 数组，预测的顶点，形状为 (44, 6890, 3)
    :param gt_vertices: numpy 数组，真实的顶点，形状为 (44, 6890, 3)
    :param smpl_model: 包含面信息的模型
    """
    stop_visualization = [False]  # 用于控制停止可视化的标志位
    next_frame = [False]  # 用于控制进入下一帧的标志位

    def close_visualizer(vis):
        stop_visualization[0] = True
        vis.close()
        return False

    def next_frame_callback(vis):
        next_frame[0] = True
        vis.close()
        return False
    
    pred_vertices = pred['pred_vertices']
    gt_vertices = gt['gt_vertices']

    
    pred_transl = pred['global_t'].reshape(44,1,3)
    gt_transl = gt['transl'].reshape(44,1,3)

    pred_joints = pred['pred_joints'][:,:24,:]
    gt_joints = gt['gt_3D_joints']

    pred_pelvis = ((pred_joints[:,2,:] + pred_joints[:,3,:])/2) .reshape(44,1,3)
    gt_pelvis = ((gt_joints[:,2,:] + gt_joints[:,3,:])/2) .reshape(44,1,3)


    pred_vertices_align, _ = batch_compute_similarity_transform_torch(torch.from_numpy(pred_vertices), torch.from_numpy(gt_vertices))
    pred_joints_align, _ = batch_compute_similarity_transform_torch(torch.from_numpy(pred_joints), torch.from_numpy(gt_joints))
    pred_vertices_align = pred_vertices_align.numpy()
    pred_joints_align = pred_joints_align.numpy()

    mpjpe = calculate_batch_mpjpe(pred_joints-pred_transl, gt_joints-gt_transl)
    v2v = calculate_batch_v2v(pred_vertices-pred_transl, gt_vertices-gt_transl)

    pelvis_mpjpe = calculate_batch_mpjpe(pred_joints-pred_pelvis, gt_joints-gt_pelvis)
    pelvis_v2v = calculate_batch_v2v(pred_vertices-pred_pelvis, gt_vertices-gt_pelvis)

    pa_mpjpe = calculate_batch_mpjpe(pred_joints_align, gt_joints)
    pa_v2v = calculate_batch_v2v(pred_vertices_align, gt_vertices)

    


    print("------------------------ over all MPJPE and V2V -------------------------")
    print(f"from {pred_dir}:")
    print(f"MPJPE: {mpjpe['mean']}")
    print(f"V2V: {v2v['mean']}")
    print(f"pelvis_MPJPE: {pelvis_mpjpe['mean']}")
    print(f"pelvis_V2V: {pelvis_v2v['mean']}")
    print(f"PA-MPJPE: {pa_mpjpe['mean']}")
    print(f"PA-V2V: {pa_v2v['mean']}") 
    # print(pred_transl.shape, gt_transl.shape)


    i = 0
    while i < pred_vertices.shape[0]:
        if stop_visualization[0]:
            break
        next_frame[0] = False

        single_mpjpe = calculate_single_mpjpe(pred_joints[i, :, :]-pred_transl[i, :, :], gt_joints[i, :, :]-gt_transl[i, :, :])
        single_v2v = calculate_single_v2v(pred_vertices[i, :, :]-pred_transl[i, :, :], gt_vertices[i, :, :]-gt_transl[i, :, :])

        
        single_pa_mpjpe = calculate_single_mpjpe(pred_joints_align[i, :, :], gt_joints[i, :, :])
        single_pa_v2v = calculate_single_v2v(pred_vertices_align[i, :, :], gt_vertices[i, :, :])
        
        print(f"------------------------ MPJPE and V2V of sample[{i}] -------------------------")
        print(f"Single MPJPE: {single_mpjpe}")
        print(f"Single V2V: {single_v2v}")
        
        print(f"Single PA-MPJPE: {single_pa_mpjpe}")
        print(f"Single PA-V2V: {single_pa_v2v}")


        # pred_wireframe = visualize_wireframe(pred_vertices[i, :, :], smpl_model.faces, color=[0.8, 0, 0])  # 红色表示预测
        # gt_wireframe   = visualize_wireframe(gt_vertices[i, :, :], smpl_model.faces, color=[0, 0.8, 0])  # 绿色表示真实值
        
        shift1 = np.array([1.5,0,0]).reshape(1,3)
        pred_wireframe = visualize_wireframe(pred_vertices[i, :, :], smpl_model.faces, color=[0.8, 0, 0])  # 红色表示预测

        pred_wireframe_shift = visualize_wireframe(pred_vertices_align[i, :, :] + shift1, smpl_model.faces, color=[0, 0.4, 0.8])  # 红色表示预测
        gt_wireframe_shift   = visualize_wireframe(gt_vertices[i, :, :]+ shift1, smpl_model.faces, color=[0, 0.8, 0])  # 绿色表示真实值


        gt_wireframe   = visualize_wireframe(gt_vertices[i, :, :], smpl_model.faces, color=[0, 0.8, 0])  # 绿色表示真实值

        pred_compare = visualize_wireframe(pred_vertices[i, :, :]-pred_transl[i, :, :] + gt_transl[i, :, :] + 2*shift1, smpl_model.faces, color=[0, 0.4, 0.8])
        gt_compare   = visualize_wireframe(gt_vertices[i, :, :] + 2*shift1, smpl_model.faces, color=[0, 0.8, 0])

        # print(f"gt_transl = {gt_transl[i, :, :]}, pred_transl = {pred_transl[i, :, :]}")
        # print(f"gt_transl - pred_transl = {gt_transl[i, :, :] - pred_transl[i, :, :]}")

        # 创建一个坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # 创建一个可视化窗口
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f'SMPL {i}, gt[green] pred[red]')

        vis.add_geometry(pred_wireframe)
        vis.add_geometry(gt_wireframe)
        vis.add_geometry(gt_wireframe_shift)
        vis.add_geometry(pred_wireframe_shift)
        vis.add_geometry(pred_compare)
        vis.add_geometry(gt_compare)

        vis.add_geometry(coord_frame)
        
        # 按下 'Q' 键退出可视化并停止循环
        vis.register_key_callback(ord('Q'), close_visualizer)

        # 按下空格键进入下一帧
        vis.register_key_callback(ord(' '), next_frame_callback)

        vis.run()
        vis.destroy_window()

        if next_frame[0]:
            i += 1

import numpy as np
import smplx
from common import constants

smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl")
pred_dir = r'E:\WorkSpace\inbed_pose_repos\CLIFF\slp_sample\cliff_defaultweight_100epoch.npz'
data_gt = np.load(r'E:\WorkSpace\inbed_pose_repos\CLIFF\slp_sample\p102.npz')
data_pred = np.load(pred_dir)
print(data_gt.keys(), data_pred.keys())
# for k,v in data_gt.items():
#     print(k,v.shape)
# for k,v in data_pred.items():
#     print(k,v.shape)





# 调用对比函数
compare_vertices(data_pred, data_gt,smpl_model)

