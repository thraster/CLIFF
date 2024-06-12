import os
import os.path as osp
import cv2
import torch
import numpy as np
from tqdm import tqdm
import pyrealsense2 as rs
import smplx
from torch.utils.data import DataLoader
import torchgeometry as tgm
import argparse
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from models.cliff_res50.cliff import CLIFF as cliff_res50
from common import constants
from common.utils import strip_prefix_if_present, cam_crop2full
from common.renderer_pyrd import Renderer
from lib.yolov3_detector import HumanDetector
from common.mocap_dataset import MocapDataset
from lib.yolov3_dataset import DetectionDataset

from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def click_and_crop(event, x, y, flags, param):
    # 定义全局变量
    global refPt, cropping

    # 如果按下左键，则开始选择ROI区域
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # 如果释放左键，则结束选择ROI区域
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        # 绘制矩形框并显示
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("Starting RealSense camera stream...")

    # Create the model instance
    cliff = eval("cliff_" + args.backbone)
    cliff_model = cliff(constants.SMPL_MEAN_PARAMS).to(device)
    # Load the pretrained model
    print("Load the CLIFF checkpoint from path:", args.ckpt)
    state_dict = torch.load(args.ckpt)['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    # Setup the SMPL model
    smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)

    human_detector = HumanDetector()

    plt.ion()
    connections = [(12, 15), (12, 14), (12, 13), 
                   (13, 16), (16, 18), (18, 20), (20, 22), 
                   (14, 17), (17, 19), (19, 21),(21, 23), 
                   (14, 9), (13, 9), (9, 6), (6, 3), (3, 0), (0, 2), (0, 1),
                   (1, 4), (4, 7), (7, 10),
                   (2, 5), (5, 8), (8, 11)]  # 示例连接信息
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置X,y,z轴的范围
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0.2, -0.6)

    # 设置X坐标轴的单位刻度
    ax.xaxis.set_major_locator(MultipleLocator(0.2))  # 在这里设置单位刻度的值

    # 设置Y坐标轴的单位刻度
    ax.yaxis.set_major_locator(MultipleLocator(0.2))  # 在这里设置单位刻度的值

    # 设置Z坐标轴的单位刻度
    ax.zaxis.set_major_locator(MultipleLocator(0.2))  # 在这里设置单位刻度的值
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # # 调整轴线位置
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # ax.zaxis.set_ticks_position('back')
    # 调整初始视角
    ax.view_init(elev=-120, azim=-90)  # 设置俯视角为30度，方位角为45度
    # 设置三个坐标轴的比例
    ax.set_box_aspect([1.5, 2, 0.8])

    # 手动划定ROI
    print("Please select the ROI and press Enter")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    color_image = np.rot90(color_image, k=1).copy()

    roi = cv2.selectROI("ROI Selector", color_image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("ROI Selector")

    x, y, w, h = roi

    print("start detecting...")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        color_image = np.rot90(color_image, k= 1).copy() 
        roi_image = color_image[y:y+h, x:x+w]
        orig_img_bgr_all = [roi_image]

        # 创建检测数据集和数据加载器

        # 用于将原始图像列表 orig_img_bgr_all 转换为适合检测器输入的数据集。
        detection_dataset = DetectionDataset(orig_img_bgr_all, human_detector.in_dim)

        # 用于从数据集中加载数据
        detection_data_loader = DataLoader(detection_dataset, batch_size=args.batch_size, num_workers=0)

        detection_all = []

        for batch_idx, batch in enumerate(detection_data_loader):
            # print("index: ",batch_idx)
            norm_img = batch["norm_img"].to(device).float()
            dim = batch["dim"].to(device).float()
            detection_result = human_detector.detect_batch(norm_img, dim)
            detection_result[:, 0] += batch_idx * args.batch_size
            detection_all.extend(detection_result.cpu().numpy())

        detection_all = np.array(detection_all)

        if len(detection_all) == 0:
            print("no people detected in roi", end='\r')
            # print("num of people: ",len(detection_all))


        mocap_db = MocapDataset(orig_img_bgr_all, detection_all)
        mocap_data_loader = DataLoader(mocap_db, batch_size=args.batch_size, num_workers=0)

        for batch in mocap_data_loader:
            norm_img = batch["norm_img"].to(device).float()
            center = batch["center"].to(device).float()
            scale = batch["scale"].to(device).float()
            img_h = batch["img_h"].to(device).float()
            img_w = batch["img_w"].to(device).float()
            focal_length = batch["focal_length"].to(device).float()

            cx, cy, b = center[:, 0], center[:, 1], scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
            bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
            bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)

            full_img_shape = torch.stack((img_h, img_w), dim=-1)
            pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)

            pred_output = smpl_model(betas=pred_betas,
                                     body_pose=pred_rotmat[:, 1:],
                                     global_orient=pred_rotmat[:, [0]],
                                     pose2rot=False,
                                     transl=pred_cam_full,
                                     return_full_pose=True)
            # pred_vertices = pred_output.vertices.cpu()
            _joints = pred_output.joints.cpu().squeeze(0)[0:24,:] # 筛选出前24个原始关节点，不要面部等关节点
            
            pred_psoe = pred_output.body_pose.cpu().squeeze(0)
            # print("joints: ",pred_joints.shape)
            # print("pose: ",pred_psoe.shape)
            # Visualization code here (optional)
                    # 绘制3D关节点

            _joints[:,2] = -_joints[:,2]
            pred_joints = _joints - _joints[0,:]
            

            for line in ax.lines[:]:
                 line.remove()
            for scatter in ax.collections[:]:
                scatter.remove()
            for text in ax.texts[:]:
                text.remove()

            ax.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2], c='b', marker='o', label='pred_joints')
            
            for connection in connections:
                joint1, joint2 = connection
                x1, y1, z1 = pred_joints[joint1]
                x2, y2, z2 = pred_joints[joint2]
                ax.plot([x1, x2], [y1, y2], [z1, z2], c='r')

            cv2.imshow("RGB", color_image)

            plt.pause(0.5)  # 暂停一段时间，不然画的太快会卡住显示不出来

            plt.ioff()  # 关闭画图窗口




            if args.viz:
                renderer = Renderer(resolution=(640, 480))
                rendered_img = renderer(pred_vertices, pred_cam_full)
                cv2.imshow('RealSense', rendered_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args.save_results:
            result_filepath = "realsense_cliff_results.npz"
            np.savez(result_filepath, pose=pred_pose.cpu().numpy(),
                     shape=pred_betas.cpu().numpy(), global_t=pred_cam_full.cpu().numpy(),
                     pred_joints=pred_output.joints.cpu().numpy(), focal_l=focal_length.cpu().numpy(),
                     detection_all=detection_all)

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_type', default='realsense', choices=['realsense'],
                        help='input type')
    parser.add_argument('--ckpt', default="data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt",
                        help='path to the pretrained checkpoint')
    parser.add_argument("--backbone", default="hr48", choices=['res50', 'hr48'],
                        help="the backbone architecture")
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for detection and motion capture')
    parser.add_argument('--viz', action='store_true',
                        help='visualize and output the mesh')
    parser.add_argument('--save_results', action='store_true',
                        help='save the results as a npz file')
    parser.add_argument('--pose_format', default='aa', choices=['aa', 'rotmat'],
                        help='aa for axis angle, rotmat for rotation matrix')

    args = parser.parse_args()
    main(args)



















#     pred_vert_arr = []
#     if args.save_results:
#         smpl_pose = []
#         smpl_betas = []
#         smpl_trans = []
#         smpl_joints = []
#         cam_focal_l = []

#     mocap_db = MocapDataset(orig_img_bgr_all, detection_all)
#     mocap_data_loader = DataLoader(mocap_db, batch_size=min(args.batch_size, len(detection_all)), num_workers=0)
#     for batch in tqdm(mocap_data_loader):
#         norm_img = batch["norm_img"].to(device).float()
#         center = batch["center"].to(device).float()
#         scale = batch["scale"].to(device).float()
#         img_h = batch["img_h"].to(device).float()
#         img_w = batch["img_w"].to(device).float()
#         focal_length = batch["focal_length"].to(device).float()

#         cx, cy, b = center[:, 0], center[:, 1], scale * 200
#         bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
#         # The constants below are used for normalization, and calculated from H36M data.
#         # It should be fine if you use the plain Equation (5) in the paper.
#         bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
#         bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

#         with torch.no_grad():
#             pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)

#         # convert the camera parameters from the crop camera to the full camera
#         full_img_shape = torch.stack((img_h, img_w), dim=-1)
#         pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)

#         pred_output = smpl_model(betas=pred_betas,
#                                  body_pose=pred_rotmat[:, 1:],
#                                  global_orient=pred_rotmat[:, [0]],
#                                  pose2rot=False,
#                                  transl=pred_cam_full)
#         pred_vertices = pred_output.vertices
#         pred_vert_arr.extend(pred_vertices.cpu().numpy())

#         if args.save_results:
#             if args.pose_format == "aa":
#                 rot_pad = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
#                 rot_pad = rot_pad.expand(pred_rotmat.shape[0] * 24, -1, -1)
#                 rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad), dim=-1)
#                 pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)  # N*72
#             else:
#                 pred_pose = pred_rotmat  # N*24*3*3

#             smpl_pose.extend(pred_pose.cpu().numpy())
#             smpl_betas.extend(pred_betas.cpu().numpy())
#             smpl_trans.extend(pred_cam_full.cpu().numpy())
#             smpl_joints.extend(pred_output.joints.cpu().numpy())
#             cam_focal_l.extend(focal_length.cpu().numpy())

#     if args.save_results:
#         print(f"Save results to \"{result_filepath}\"")
#         np.savez(result_filepath, imgname=img_path_list,
#                  pose=smpl_pose, shape=smpl_betas, global_t=smpl_trans,
#                  pred_joints=smpl_joints, focal_l=cam_focal_l,
#                  detection_all=detection_all)
#         np.save("smpl_v_cliff_slp", np.array(pred_vert_arr).reshape(6890, 3))




# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--input_type', default='folder', choices=['image', 'folder', 'video'],
#                         help='input type')
#     parser.add_argument('--input_path', default='inbed_samples', help='path to the input data')

#     parser.add_argument('--ckpt',
#                         default="data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt",
#                         help='path to the pretrained checkpoint')
#     parser.add_argument("--backbone", default="hr48", choices=['res50', 'hr48'],
#                         help="the backbone architecture")
#     parser.add_argument('--batch_size', type=int, default=32,
#                         help='batch size for detection and motion capture')
    
#     parser.add_argument('--viz', action='store_true',
#                         help='visulazing and output the mesh')
    
#     parser.add_argument('--save_results', action='store_true',
#                         help='save the results as a npz file')
#     parser.add_argument('--pose_format', default='aa', choices=['aa', 'rotmat'],
#                         help='aa for axis angle, rotmat for rotation matrix')

#     parser.add_argument('--show_bbox', action='store_true',
#                         help='show the detection bounding boxes')
#     parser.add_argument('--show_sideView', action='store_true',
#                         help='show the result from the side view')

#     parser.add_argument('--make_video', action='store_true',
#                         help='make a video of the rendering results')
#     parser.add_argument('--frame_rate', type=int, default=30, help='frame rate')

#     args = parser.parse_args()
#     main(args)

# # python demo.py --input_type image --input_path test_samples/136.png --save_results --show_bbox --show_sideView