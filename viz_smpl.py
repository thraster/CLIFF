
import smplx
import open3d as o3d
import numpy as np



# 创建一个空的可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720)

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
vis.add_geometry(coord_frame)

# # 创建平面的顶点和三角形信息
# vertices = [[0, 0, 0], [0, 0.9, 0], [2, 0, 0], [2, 0.9, 0]]
# triangles = [[0, 2, 1], [1, 2, 3]] # 调整顶点的顺序 使得两个三角形的法线指向屏幕外部

# # 创建TriangleMesh对象
# plane = o3d.geometry.TriangleMesh()
# plane.vertices = o3d.utility.Vector3dVector(vertices)
# plane.triangles = o3d.utility.Vector3iVector(triangles)

# # 设置平面的颜色
# plane.paint_uniform_color([0.5, 0.7, 0.3])  # 设置平面的颜色为绿色

# # 添加平面到可视化窗口
# vis.add_geometry(plane)

SMPL_MODEL_DIR = r'E:\WorkSpace\inbed_pose_repos\CLIFF\data'

smpl_model = smplx.create(SMPL_MODEL_DIR, "smpl")

pcd = o3d.io.read_point_cloud("filtered_point_cloud.ply")
vis.add_geometry(pcd)
# 加载SMPL模型的顶点数据1
vertices1 = np.load('smpl_v_cliff_slp.npy')
# vertices1 = vertices1 - vertices1[0,:]
# 创建TriangleMesh对象并设置顶点和面数据
mesh1 = o3d.geometry.TriangleMesh()
mesh1.vertices = o3d.utility.Vector3dVector(vertices1)
mesh1.triangles = o3d.utility.Vector3iVector(smpl_model.faces)
mesh1.compute_vertex_normals()
mesh1.paint_uniform_color([1, 0.5, 0]) # 黄色

# 加载SMPL模型的顶点数据2
vertices2 = np.load('smpl_v_gt_slp.npy')
# vertices2 = vertices2 - vertices2[0,:]
# 创建TriangleMesh对象并设置顶点和面数据
mesh2 = o3d.geometry.TriangleMesh()
mesh2.vertices = o3d.utility.Vector3dVector(vertices2)
mesh2.triangles = o3d.utility.Vector3iVector(smpl_model.faces)
mesh2.compute_vertex_normals()
mesh2.paint_uniform_color([0, 0.5, 1]) # 蓝色

# o3d.visualization.draw_geometries([mesh1])


# o3d.visualization.draw_geometries([mesh2])
print("mesh1: ",vertices1[0,:])
print("mesh2: ",vertices2[0,:])
# 添加第一个mesh到可视化窗口
vis.add_geometry(mesh1)

# 添加第二个mesh到可视化窗口
vis.add_geometry(mesh2)

# 渲染和显示模型
vis.run()

# 关闭可视化窗口
vis.destroy_window()
