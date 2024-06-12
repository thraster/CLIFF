import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from smplx import SMPL
import open3d as o3d

# 加载 SMPL 模型
neutral_path = r'E:\WorkSpace\inbed_pose_repos\CLIFF\data\smpl\SMPL_NEUTRAL.pkl'  # 替换为你的 SMPL 模型路径
smpl_neutral = SMPL(model_path=neutral_path, gender='neutral')

# 生成随机形状和姿态参数
betas = torch.randn(1, 10) / 10  # 随机形状参数
pose = torch.randn(1, 72) / 10   # 随机姿态参数

def create_joints(smpl_model, betas, pose, translation):
    output = smpl_model(betas=betas, body_pose=pose[:, 3:], global_orient=pose[:, :3])
    vertices = output.vertices.detach().cpu().numpy().squeeze()  # 顶点信息
    vertices += translation  # 平移顶点位置以便可视化
    joint = output.joints.detach().cpu().numpy().squeeze()  # 关节信息
    joints = joint + translation  # 平移关节位置以便可视化
    faces = smpl_model.faces  # SMPL 模型的面片信息

    return joints

joints = create_joints(smpl_neutral, betas, pose, np.array([0, 0, 0]))[:24,:]


# 假设 SLPL 到 SMPL 的映射表如下
slp_to_smpl_mapping = {
    0: 8,
    1: 5,
    2: 2,
    3: 1,
    4: 4,
    5: 7,
    6: 21,
    7: 19,
    8: 17,
    9: 16,
    10: 19,
    11: 20,
    12: 12,
    13: 15
}

# 使用映射表提取对应关节点
mapped_joints = joints[list(slp_to_smpl_mapping.values())]

# 创建一个更大的图形窗口
fig = plt.figure(figsize=(18, 6))

# 原始关节点的3D可视化
ax1 = fig.add_subplot(141, projection='3d')
ax1.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='blue', marker='o', label='Original Joints')
for i, (x, y, z) in enumerate(joints):
    ax1.text(x, y, z, str(i), color='red', fontsize=8)
ax1.set_title('Original Joint Visualization (3D)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 映射后的关节点的3D可视化
ax2 = fig.add_subplot(142, projection='3d')
ax2.scatter(mapped_joints[:, 0], mapped_joints[:, 1], mapped_joints[:, 2], c='green', marker='o', label='Mapped Joints')
for i, (x, y, z) in enumerate(mapped_joints):
    ax2.text(x, y, z, str(list(slp_to_smpl_mapping.keys())[i]), color='black', fontsize=8)
ax2.set_title('Mapped Joint Visualization (3D)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# 原始关节点的2D投影可视化
ax3 = fig.add_subplot(143)
ax3.scatter(joints[:, 0], joints[:, 1], c='blue', marker='o', label='Original Joints 2D')
for i, (x, y) in enumerate(joints[:, :2]):
    ax3.text(x, y, str(i), color='red', fontsize=8)
ax3.set_title('Original Joint Visualization (2D)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')

# 映射后的关节点的2D投影可视化
ax4 = fig.add_subplot(144)
ax4.scatter(mapped_joints[:, 0], mapped_joints[:, 1], c='green', marker='o', label='Mapped Joints 2D')
for i, (x, y) in enumerate(mapped_joints[:, :2]):
    ax4.text(x, y, str(list(slp_to_smpl_mapping.keys())[i]), color='black', fontsize=8)
ax4.set_title('Mapped Joint Visualization (2D)')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')

plt.show()