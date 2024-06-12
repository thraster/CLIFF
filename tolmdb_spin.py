'''
处理SLP数据集和SLP-3DFIT的标注 同时生成适合CLIFF训练用的额外的bbox info等信息, 最终打包为lmdb

gt_keypoints_2d = batch['keypoints']    # 2D keypoints
gt_pose = batch['pose']                 # SMPL pose parameters
gt_betas = batch['betas']               # SMPL beta parameters
gt_joints = batch['joints_3d']          # 3D SMPL 关节点

norm_img = batch["norm_img"].float()    # 归一化的 RGB 图像
center = batch["center"].float()        # bbox的中心点位置
scale = batch["scale"].float()          # bbox的scale
img_h = batch["img_h"].float()          # 图像高度
img_w = batch["img_w"].float()          # 图像宽度
focal_length = batch["focal_length"].float() # 图像拍摄时的焦距


1. 读取对应的slp-3dfit信息 获得pose beta joints

2. 加载RGB图像 使用yolo检测出人体的bbox 送入mocapdataset类 获得norm_img center scale img_h img_w

3. 自己处理 获得2d keypoints和focal_length


# appedix
1. SLP数据集的相机信息
    RGB:  
    f_R = [902.6, 877.4]; 
    c_R= [278.4,  525,1];

    depth: 
    c_d = [208.1, 259.7];     
    f_d = [367.8, 367.8];


'''

import os
import pickle
import lmdb
import cv2
# from smpl_torch import SMPLModel
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.yolov3_detector import HumanDetector
from common.mocap_dataset import MocapDataset
from lib.yolov3_dataset import DetectionDataset
from smplx import SMPL
import scipy.io

device = 'cuda'

# model_path_f = r'E:\WorkSpace\inbed-pose\pose_master\smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl'
# smpl_f = SMPLModel(device=device,model_path=model_path_f).to(device)

# model_path_m = r'E:\WorkSpace\inbed-pose\pose_master\smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
# smpl_m = SMPLModel(device=device,model_path=model_path_m).to(device)


# 指定包含.pkl文件的输入文件夹路径
pkl_folder = r'E:\WorkSpace\inbed_pose_repos\SLP-3Dfits\dataset\fits\train'  # 替换为实际的输入文件夹路径

img_folder = r'E:\WorkSpace\dataset\SLP\danaLab'
# 指定LMDB数据集的输出路径
output_path = r'E:\WorkSpace\inbed_pose_repos\SPIN\datasets\train_slp4cliff'  # 替换为LMDB数据集的输出路径

# 加载 SMPL 模型
neutral_path = r'E:\WorkSpace\inbed_pose_repos\CLIFF\data\smpl\SMPL_NEUTRAL.pkl'  # 替换为你的 SMPL 模型路径
smpl_neutral = SMPL(model_path=neutral_path, gender='neutral').to(device)

male_path = r'E:\WorkSpace\inbed_pose_repos\CLIFF\data\smpl\SMPL_MALE.pkl'  # 替换为你的 SMPL 模型路径
smpl_male = SMPL(model_path=male_path, gender='male').to(device)

female_path = r'E:\WorkSpace\inbed_pose_repos\CLIFF\data\smpl\SMPL_female.pkl'  # 替换为你的 SMPL 模型路径
smpl_female = SMPL(model_path=female_path, gender='female').to(device)
human_detector = HumanDetector()


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    # print('points: ',points[i,i,:])
    # print('translation: ',translation[0])
    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def create_lmdb_dataset(pkl_folder, img_folder, output_path):
    # 创建LMDB环境
    env = lmdb.open(output_path, map_size= 1024 * 1024 * 10000 )# 1000MB大小
    key = 0
    # 遍历输入文件夹中的文件
    for root, dirs, files in os.walk(pkl_folder):
        print(f"处理文件: {root}")
        for f in files:
            if f.endswith('.pkl'):
                # print(int(f[1:-4]))
                data = {}

                file_path = os.path.join(root, f)
                # print(file_path)
                RGB_path = img_folder + r'\00' + file_path[-11:-8] + '\RGB' + r'\uncover' + '\image_0000' + f[1:-4] +'.png'
                DEPTH_path = img_folder + r'\00' + file_path[-11:-8] + '\depth' + r'\uncover' + '\image_0000' + f[1:-4] +'.png'
                PMAT_path = img_folder + r'\00' + file_path[-11:-8] + '\PM' + r'\uncover' + '\image_0000' + f[1:-4] +'.png'

                # 把SLP-3DFIT的标注投影到RGB 2D
                PTr_depth = np.load(img_folder + r'\00' + file_path[-11:-8] + r'\align_PTr_depth.npy')
                PTr_RGB = np.load(img_folder + r'\00' + file_path[-11:-8] + r'\align_PTr_RGB.npy')
                PTr_D2RGB = np.dot(np.linalg.inv(PTr_RGB), PTr_depth)
                PTr_D2RGB = PTr_D2RGB / np.linalg.norm(PTr_D2RGB)

                data['keypoints'] =  torch.from_numpy(scipy.io.loadmat(img_folder + r'\00' + file_path[-11:-8]+r'\joints_gt_RGB.mat')['joints_gt'][:,:,int(f[1:-4])-1]).float()
                # print(img_path)

                # 读取pickle文件
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                
                # data['RGB'] = torch.from_numpy(cv2.imread(RGB_path, cv2.IMREAD_GRAYSCALE))
                
                data['RGB'] = cv2.imread(RGB_path)
                data['DEPTH'] = torch.from_numpy(cv2.imread(DEPTH_path, cv2.IMREAD_GRAYSCALE))
                data['PMAT'] = torch.from_numpy(cv2.imread(PMAT_path, cv2.IMREAD_GRAYSCALE))
                data['focal_length'] = torch.tensor([890.])
                data['DEPTH_focal_length'] = torch.from_numpy(content['camera_focal_length_x']).unsqueeze(0)
                data['DEPTH_transl'] = torch.from_numpy(content['transl']).unsqueeze(0)
                data['betas'] = torch.from_numpy(content['betas']).unsqueeze(0)
                # data['pose'] = torch.from_numpy(content['body_pose'])

                data['pose'] = torch.cat((torch.from_numpy(content['global_orient']), 
                                          torch.from_numpy(content['body_pose'])), dim=0).unsqueeze(0)
                
                if content['gender'] == 'male':

                    data['gender'] = torch.tensor(1).unsqueeze(0).to(torch.uint8)

                    output = smpl_male(
                                            body_pose = data['pose'][:,3:].to(device),
                                            betas = data['betas'].to(device),
                                            global_orient=data['pose'][:,:3].to(device)
                                        )
                    
                    joints = output.joints.detach().cpu().squeeze()  # 关节信息
                    # print(joints.shape)
                    data['joints_3d'] = joints[:24,:]
                    # print(data['joints_3d'].unsqueeze(0).shape)
                    # print(torch.from_numpy(content['camera_rotation']).unsqueeze(0).shape)
                    # print(data['DEPTH_transl'].shape)
                    # print(data['DEPTH_focal_length'].shape)
                    # print(torch.tensor([208.1, 259.7]).shape)

                    gt_keypoints_2d = perspective_projection(points=data['joints_3d'].unsqueeze(0),
                                      rotation=torch.from_numpy(content['camera_rotation']).unsqueeze(0),
                                    #   translation=torch.zeros([1,3]),
                                      translation= data['DEPTH_transl'],
                                      focal_length= data['DEPTH_focal_length'],
                                      camera_center= torch.tensor([208.1, 259.7]) #- torch.tensor([0, 15])
                                      )
                    # print(PTr_D2RGB.shape)
                    # print(gt_keypoints_2d[:, :, :2].shape)

                    data['joints_2d'] = np.array(list(
                        map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_D2RGB)[0], gt_keypoints_2d[:, :, :2].numpy())
                    )).squeeze(0)


                elif content['gender'] == 'female':

                    data['gender'] = torch.tensor(0).unsqueeze(0).to(torch.uint8)

                    output = smpl_female(
                                            body_pose = data['pose'][:,3:].to(device),
                                            betas = data['betas'].to(device),
                                            global_orient=data['pose'][:,:3].to(device)
                                        )
                    # pose = data['pose'].squeeze(0)[:,3:],
                    # betas = data['betas'].squeeze(0),
                    # global_orient=data['pose'].squeeze(0)[:,:3]
                    joints = output.joints.detach().cpu().squeeze()  # 关节信息
                    # print(joints.shape)
                    data['joints_3d'] = joints[:24,:]
                    # print(data['joints_3d'].unsqueeze(0).shape)
                    # print(torch.from_numpy(content['camera_rotation']).unsqueeze(0).shape)
                    # print(data['DEPTH_transl'].shape)
                    # print(data['DEPTH_focal_length'].shape)
                    # print(torch.tensor([208.1, 259.7]).shape)

                    gt_keypoints_2d = perspective_projection(points=data['joints_3d'].unsqueeze(0),
                                      rotation=torch.from_numpy(content['camera_rotation']).unsqueeze(0),
                                    #   translation=torch.zeros([1,3]),
                                      translation= data['DEPTH_transl'],
                                      focal_length= data['DEPTH_focal_length'],
                                      camera_center= torch.tensor([208.1, 259.7]) #- torch.tensor([0, 15])
                                      )
                    # print(PTr_D2RGB.shape)
                    # print(gt_keypoints_2d[:, :, :2].shape)

                    data['joints_2d'] = np.array(list(
                        map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_D2RGB)[0], gt_keypoints_2d[:, :, :2].numpy())
                    )).squeeze(0)

                output = smpl_neutral(
                                            body_pose = data['pose'][:,3:].to(device),
                                            betas = data['betas'].to(device),
                                            global_orient=data['pose'][:,:3].to(device)
                                        )
                joints = output.joints.detach().cpu().squeeze()  # 关节信息
                
                data['joints_3d_neutral'] = joints[:24,:]
                gt_keypoints_2d = perspective_projection(points=data['joints_3d_neutral'].unsqueeze(0),
                                      rotation=torch.from_numpy(content['camera_rotation']).unsqueeze(0),
                                    #   translation=torch.zeros([1,3]),
                                      translation= data['DEPTH_transl'],
                                      focal_length= data['DEPTH_focal_length'],
                                      camera_center= torch.tensor([208.1, 259.7]) #- torch.tensor([0, 15])
                                      )
                data['joints_2d_neutral'] = np.array(list(
                    map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_D2RGB)[0], gt_keypoints_2d[:, :, :2].numpy())
                )).squeeze(0)
                
                orig_img_bgr_all = [data['RGB']]
                detection_dataset = DetectionDataset(orig_img_bgr_all, human_detector.in_dim)
                detection_data_loader = DataLoader(detection_dataset, batch_size=1, num_workers=0)
                detection_all = []
                
                for batch_idx, batch in enumerate(detection_data_loader):
                    norm_img = batch["norm_img"].to(device).float()
                    dim = batch["dim"].to(device).float()

                    detection_result = human_detector.detect_batch(norm_img, dim)
                    detection_result[:, 0] += batch_idx
                    detection_all.extend(detection_result.cpu().numpy())
                detection_all = np.array(detection_all)

                if len(detection_all) != 1:
                    print(f"detected {len(detection_all)} people! \n at [{f}]")
                    print(f"skipping...")
                    continue

                mocap_db = MocapDataset(orig_img_bgr_all, detection_all)
                mocap_data_loader = DataLoader(mocap_db, batch_size=1, num_workers=0)

                for batch in mocap_data_loader:
                    data['norm_image'] = batch["norm_img"].float()
                    data['center'] = batch["center"].float()
                    data['scale'] = batch["scale"].float()
                    data['img_h'] = batch["img_h"].float()
                    data['img_w'] = batch["img_w"].float()
                

                data['DEPTH_transl'] = data['DEPTH_transl'].squeeze(0)
                data['betas'] = data['betas'].squeeze(0)
                data['pose'] = data['pose'].squeeze(0)
                try:
                    data['norm_image'] = data['norm_image'].squeeze(0)
                except:
                    cv2.imshow('RGB',data['RGB'])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                data['center'] = data['center'].squeeze(0)

                # for k,v in data.items():
                #     print(f"{k}: shape = {v.shape}, dtype = {v.dtype}")
                    
                    # print(key)
                # data['skeleton'] = data['skeleton'].unsqueeze(0)
                # # data['image'] = data['image'].unsqueeze(0)
                # data['shape'] = data['shape'].unsqueeze(0)
                # data['trans'] = data['trans'].unsqueeze(0)
                # data['pose'] = data['pose'].unsqueeze(0)

                # for key, value in data.items():
                #     print(f"{key} : {value.shape}")
                # break
                # 将数据转换为字节
                data_bytes = pickle.dumps(data)
                
                # 为LMDB条目生成唯一键
                

                # 将数据写入LMDB数据库
                with env.begin(write=True) as txn:
                    txn.put(str(key).encode('ascii'), data_bytes)
                key += 1
    print(f"输出位置: {output_path}")

    # 关闭LMDB环境
    env.close()



# 创建LMDB数据集
create_lmdb_dataset(pkl_folder, img_folder, output_path)