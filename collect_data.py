import pyrealsense2 as rs
import numpy as np
import cv2
import ctypes
import numpy as np
from time import sleep
from enum import Enum
import sys
from io import StringIO
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from IPython import display
import time
import pickle
import datetime
import os
import math

'''
Yotlive SDK part
'''

class ConnectionMode(Enum):
    UNKOWN = -1
    Ethernet = 20
    WifiAp = 21
    Serial = 22

class ConnectionStatus(Enum):
    Closed = 0
    Opened = 1
    Connected = 2
    ToSample = 3
    Sampling = 4
    ToStop = 5
    Stopped = 6

OnSamplingType = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int))
OnDeviceReadyType = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_double, ctypes.c_uint32)
OnStatusChangedType = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)
OnDeviceRemovedType = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32)
OnErrorType = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p)

class PyCbkBundle(ctypes.Structure):
    _fields_ = [
        ('onSampling', OnSamplingType),
        ('onDeviceReady', OnDeviceReadyType),
        ('onStatusChanged', OnStatusChangedType),
        ('onDeviceRemoved', OnDeviceRemovedType),
        ('onError', OnErrorType),
    ]

connectionMap = ({})
# row = 24
# col = 48



@OnSamplingType
def onSampling(code, rows, cols, data):
    global redirected_arr
    arr = np.ctypeslib.as_array(data, (rows, cols))
    redirected_arr = arr.T # 转置为48*24的形式, 因为模型处理的都是竖屏的输入




@OnDeviceReadyType
def onDeviceReady(connId, code, rows, cols, senselArea, maxPressure):
    #保存起来，可以根据connId查询相关信息
    connectionMap[connId] = {
        'code': code,
        'rows': rows,
        'cols': cols,
        'area': senselArea,
        'max': maxPressure
    }
    print(f"\n-------Yotlive pressure sensor is ready------\nconnection id: {connId}\ncode: {code}\n"
          f"size: {rows} x {cols}\nsensel area: {senselArea}\n"
          f"calibrated max pressure: {maxPressure}")
    # 多数情况下，可以在onDeviceReady中启动采样，本代码通过用户根据菜单选择控制，所以不需要在这里启动
    # 以下两种调用方法都可以
    # AxisLinker.startSampling(connId, False)
    # AxisLinker.startSampling(code, True)

@OnStatusChangedType
def onStatusChanged(connId, code, newStatus, oldStatus):
    oldStatusName = ConnectionStatus(int(oldStatus)).name
    newStatusName = ConnectionStatus(int(newStatus)).name
    # print(connId, code, "status changes:", oldStatusName, "-->", newStatusName)

@OnDeviceRemovedType
def onDeviceRemoved(connId, code):
    print("device removed", connId, code)

@OnErrorType
def onError(connId, msg, errName):
    # msg/errName在动态库中是cosnt char*，到这里时bytes类型，需要按gbk解码
    global _usb
    global _wifi
    if connId == 1:
        _usb = 0
    if connId == 2:
        _wifi = 0
    print(connId, "is on error:", msg.decode("gbk"), errName.decode("gbk"))


def formatiing(data):
    rows = 24
    cols = 48
    for i in range(0, rows):
        line = []
        for j in range(0, cols):
            line.append(data[i][j])

        formatted = [' {} '.format(num) for num in line]
        print(" ".join(formatted))

class YotliveCapture:
    def __init__(self):
        self._wifi = 1
        self._usb = 1
        
        self.AxisLinker = ctypes.CDLL(r'E:\WorkSpace\inbed-pose\yotlive_sdk\AxisLinker.dll')

        self.bundle = PyCbkBundle.in_dll(self.AxisLinker, "cbkBundle")
        self.bundle.onSampling = onSampling
        self.bundle.onDeviceReady = onDeviceReady
        self.bundle.onStatusChanged = onStatusChanged
        self.bundle.onError = onError
        self.bundle.onDeviceRemoved = onDeviceRemoved

        self.connIdUsb = self.AxisLinker.addConnection(ConnectionMode.Serial.value, ctypes.c_char_p(b"COM5"), 115200)
        # self.connIdCable = self.AxisLinker.addConnection(ConnectionMode.Ethernet.value, ctypes.c_char_p(b"192.168.0.57"), 2007)
        # print("connIdUsb =", self.connIdUsb, "connIdCable =", self.connIdCable)

    def start_capture(self):
        sleep(0.5)
        self.AxisLinker.startSampling(self.connIdUsb, False)
        return True

    def stop_capture(self):
        self.AxisLinker.stopSampling(self.connIdUsb, False)

    def get_pmat(self):
        # sleep(0.5)
        # self.start_capture()
        # print(redirected_arr)
        return redirected_arr
    

class D455Capture:
    '''
    相机内参
    camera_matrix: [[640.933    0.     366.2003]
                    [  0.     641.9774 645.2603]
                    [  0.       0.       1.    ]]
dist_coeffs: [-0.05653619  0.06503928  0.0004415   0.001578   -0.02057174]
    '''
    def __init__(self, col = 640, row = 360, fps = 5):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, col, row, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, col, row, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)
        
        # 创建RealSense滤波器对象
        self.spatial_filter = rs.spatial_filter()  # 空间滤波器
        self.temporal_filter = rs.temporal_filter()  # 时间滤波器
        # self.holes_filling_filter = rs.hole_filling_filter(0) # 补洞 fill_from_left

        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
        # self.temporal_filter.set_option(rs.option.persis, 20)

        # self.spatial_filter.set_option(rs.option.holes_fill, rs.holes_filling_filter.fill_from_left)



        # 获取流配置
        self.col = col
        self.row = row
        self.fps = fps
        # 设置marker参数
        self.marker_size = 150 # 单位(mm)
        
        # 设置D455的内参 @1280*720 分辨率
        # self.camera_matrix = np.array(object=[[642.4146, 0.,       645.2603],
        #                                       [0.,       641.3695, 366.2003],
        #                                       [0.,       0.,       1.      ]])

        # self.dist_coeffs = np.array(object=[-0.05653619, 0.06503928, 0.0004415,  0.001578,  -0.02057174])

    def get_camera_intrinsic(self): 
        """
        get the camera intrinsic matrix and distortion coefficients
        """
        # Get the first frame to get the camera intrinsics
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # Get the color frame's intrinsics
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        # Define camera matrix and distortion coefficients
        self.camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx], 
                                       [0, intrinsics.fy, intrinsics.ppy], 
                                       [0, 0, 1]],
                                    dtype=np.float32)
        # 顺时针旋转90度的内参
        self.camera_matrix = np.array([[intrinsics.fy, 0, intrinsics.ppy], 
                                       [0, intrinsics.fx, intrinsics.ppx], 
                                       [0, 0, 1]],
                                    dtype=np.float32)
        self.dist_coeffs = np.array(intrinsics.coeffs)
        print(f"camera_matrix: {self.camera_matrix}")
        print(f"dist_coeffs: {self.dist_coeffs}")
        
    def start_capture(self):
        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        # Get the sensor of the color stream and set parameters
        sensor = profile.get_device().query_sensors()[1]  # Color sensor is at index 1
        # sensor.set_option(rs.option.enable_auto_exposure, 0)  # Turn off auto exposure
        sensor.set_option(rs.option.enable_auto_exposure, 1)  # Enable auto exposure
        # sensor.set_option(rs.option.exposure, 20)  # Set exposure to a 设置快门速度为625ms
        # sensor.set_option(rs.option.contrast, 50)
        # sensor.set_option(rs.option.sharpness, 80)  # 80
        sensor.set_option(rs.option.enable_auto_white_balance, 1)

        # 设置深度相机为自动曝光
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)



        # 获取内参
        self.get_camera_intrinsic()
        print(f"\n-----------Realsense D455 is ready----------")
        print(f"缩放因子: ",depth_scale)
        print(f"Depth stream: {self.col}*{self.row} @{self.fps}fps")
        print(f"RGB stream info: {self.col}*{self.row} @{self.fps}fps\n")

        
    def stop_capture(self):
        self.pipeline.stop()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        color_image = None
        depth_image = None

        if color_frame and depth_frame:
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            # 对深度图像进行滤波处理
            filtered_depth_frame = self.spatial_filter.process(depth_frame)
            filtered_depth_frame = self.temporal_filter.process(filtered_depth_frame)
            depth_image = np.asanyarray(filtered_depth_frame.get_data())

        # aruco.drawDetectedMarkers函数对输入图像的内存布局有特定的要求
        # 需要传入一个cv::Mat或 cv::UMat对象
        # 由于在np.rot90之后，图像的内存布局与OpenCV的要求不匹配，因此会导致报错
        # 解决这个问题的一种方法是创建一个副本，确保图像的内存布局符合OpenCV的要求
        # 使用np.rot90(img, k=-1).copy()来创建一个副本
        depth_image = np.rot90(depth_image, k= 1).copy()
        color_image = np.rot90(color_image, k= 1).copy() 

        return color_image, depth_image

    @staticmethod
    def get_homo_matrix(rvec, tvec):
        """get homo matrix by rvec and tvec"""
        R = cv2.Rodrigues(rvec.squeeze())[0]
        t = tvec.squeeze()

        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[:3, 3] = t.T
        
        return matrix



    # def detect_markers(img):
    #     # 定义ArUco字典
    #     aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    #     parameters = cv2.aruco.DetectorParameters_create()

    #     # 检测marker
    #     corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    #     print(corners)
    #     # 绘制检测到的marker
    #     cv2.aruco.drawDetectedMarkers(img, corners, ids)

        return img, corners, ids

    def detect_markers(self, color_image):
        '''
        检测图像中的aruco marker, 并且计算每个marker作为世界坐标系对应的外参矩阵
        '''
        # 通道交换：RGB到RBG
        # color_image = color_image[:, :, [1, 2, 0]]
        # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX # 子像素级别优化
        corners, ids, _ = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=parameters)
        # print("corners: ",corners)
        marker_ids = [id[0] for id in ids] if ids is not None else []

        tag_pose = {}

        if len(corners) > 0:
            # print(corners)
            tag_pose[f'corners'] = corners
            ids = ids.flatten()	 # Flatten the ArUCo IDs list
            for (markerCorner, markerID) in zip(corners, ids):
                # # Use cornerSubPix to refine the corners
                # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
                # gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                # cv2.cornerSubPix(gray_frame, markerCorner, (5, 5), (-1, -1), criteria)

                # Estimate pose of marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorner, self.marker_size,
                                                                        self.camera_matrix, self.dist_coeffs)
                # get the projection matrix
                pose = self.get_homo_matrix(rvecs, tvecs)
                tag_pose[f'tag{markerID}'] = pose
                tag_pose[f'tag{markerID}_rvecs'] = rvecs
                tag_pose[f'tag{markerID}_tvecs'] = tvecs
                # print(pose)

            ret = True
        else:
            # print('no tag detected!')
            ret = False
        # print(tag_pose)


        return ret, marker_ids, tag_pose
        
    def draw_axis(self, img, rvec, tvec, length, thickness=3, center_color=None):
        """
        draw the axis with more features
        """
        # 绘制坐标轴的颜色
        colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
        # 定义坐标轴的点在世界坐标系中的位置
        axisPoints = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
        # 将这些点投影到图像平面
        imgPoints, _ = cv2.projectPoints(axisPoints, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        # 将投影点的坐标转换为整数
        imgPoints = imgPoints.astype(int)
        # 绘制x轴，y轴和z轴
        # print(img.shape)
        # print(imgPoints)
        cv2.line(img, tuple(imgPoints[0][0]), tuple(imgPoints[1][0]), colors[0], thickness)
        cv2.line(img, tuple(imgPoints[0][0]), tuple(imgPoints[2][0]), colors[1], thickness)
        cv2.line(img, tuple(imgPoints[0][0]), tuple(imgPoints[3][0]), colors[2], thickness)

        if center_color:
            cv2.circle(img, tuple(imgPoints[0][0]), 3, center_color, -1)

        return img




def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取像素值
        pixel_value = param[y, x]
        # print(f"shape: {param.shape}")
        print(f"Pixel Value at ({x}, {y}):", pixel_value)


def weight_count(pmat):
    # 粗略估算重量
    weight = pmat.sum() * 1.3595 * 3.24
    # print(f"total weight = {weight} g")
    return weight

def square_count(pmat):
    count = np.sum(pmat > 0)
    # print(f"usage = {count / (24*48)} %")
    # print("Acting sensor points: ",count)
    return count


def motion_detection(frame1, frame2, threshold=50):
    '''
    使用差分方法监测是否有变化
    '''
    # 将帧转换为灰度图像
    try:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    except:
        gray1 = frame1.astype(np.uint8)
        gray2 = frame2.astype(np.uint8)
        # print("gray1 shape:", gray1.shape, "dtype:", gray1.dtype)
        # print("gray2 shape:", gray2.shape, "dtype:", gray2.dtype)

    # if gray1 == gray2:
    #     print(111)
    # 计算两帧之间的差异
    frame_diff = cv2.absdiff(gray1, gray2)
    # print(frame_diff)
    # 应用阈值处理，将差异图像转换为二值图像
    _, thresholded = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow("diff", thresholded)
    # 执行形态学操作，去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    
    # 计算非零像素的数量
    num_pixels = cv2.countNonZero(morphed)
    
    # 判断是否有运动
    if num_pixels > 0:
        # print("图像发生变化")
        return True
    else:
        return False
    
def get_depth_env(CAM, wait = 8):
    # 获取无人的情况下深度相机拍摄的环境信息
    start_time = time.time()  # 记录开始时间
    while True:
        _, depth_image = CAM.get_frame()
        # 截断深度图像中大于2048的值
        depth_image = np.clip(depth_image, 0, 2048).astype(np.float32)
        # print(depth_image.max(),depth_image.min())
        # 将深度图像缩放到0-255的范围并转换为8位整数
        depth_image = depth_image / 2048
        depth_image = (depth_image * 256).astype(np.uint8)
        cv2.imshow("D", depth_image)
        cv2.waitKey(100)
        if time.time()-start_time > wait:
            
            return depth_image


def get_data(cam, pmat, save_path, save_stream, desired_fps = 5):
    '''
    开始采集并保存数据
    '''
    # global 
    PMAT = pmat()
    CAM = cam()
    
    PMAT.start_capture()
    CAM.start_capture()

    current_datetime = datetime.datetime.now()
    current_date = current_datetime.date()
    current_time = int(time.time()) #时间戳

    # print("Current Date:", current_date)
    # print("Current Time:", current_time)

    save_dir = save_path + f'\\{current_date}@{current_time}'
    # depth_env = get_depth_env(CAM)
    i = 0
    total_time = 0
    print("\nstart streaming...")
    if save_stream == "Stream":
        print(f"- 采集模式: 全程录制")
    elif save_stream == "Manual":
        print(f"- 采集模式: 手动抽帧(按下s采集一帧)")
    elif save_stream == "Auto":
        print(f"- 采集模式: 自动抽帧")
    RGB_frame1, _ = CAM.get_frame()
    P_frame1 = PMAT.get_pmat()
    # RGB_frame1 = np.rot90(RGB_frame1, k=1)
    # color_image = np.rot90(color_image, k=1)
    while True:
        
        start_time = time.time()  # 记录开始时间
        color_image, depth_image = CAM.get_frame()
        RGB_frame2 = color_image
        # depth_distamce = depth_image[359, 639]
        # print(depth_image.shape,depth_image.dtype)
        # 截断深度图像中大于2048的值
        depth_image = np.clip(depth_image, 0, 2048).astype(np.float32)
        # print(depth_image.max(),depth_image.min())
        # 将深度图像缩放到0-255的范围并转换为8位整数
        depth_image = depth_image / 2048
        depth_image = (depth_image * 256).astype(np.uint8)

       

        # depth_image = cv2.medianBlur(depth_image, 5)
        pmat = PMAT.get_pmat()
        P_frame2 = pmat
        pmat = (np.clip(pmat, 0, 255)).astype(np.uint8)
        # 沿着水平方向翻转
        # pmat = np.fliplr(np.rot90(pmat,k=1))
        # weight_count(pmat)
        # square_count(pmat)
        key = cv2.waitKey(100)
        if key == ord('q'):  # 按下 'q' 键退出循环
            break
        # print(color_image.shape,color_image.dtype)
        # print(depth_image.shape,depth_image.dtype)
        # print(pmat.shape,pmat.dtype)

        '''
        RGB  (720, 1280, 3) uint8
        D    (720, 1280) uint16
        PMAT (48, 24) int32
        保存格式：
        RGB  (720, 1280, 3) uint8
        D    (720, 1280) uint8
        PMAT (48, 24) uint8
        '''

        if save_stream == 'Stream':
            os.makedirs(save_dir+'\\RGB', exist_ok=True)
            os.makedirs(save_dir+'\\DEPTH', exist_ok=True)
            os.makedirs(save_dir+'\\PMAT', exist_ok=True)
            # 保存 color_image 为 PNG 格式
            cv2.imwrite(save_dir+f'\\RGB\\{i}.png', color_image)

            # 保存 depth_image 为 PNG 格式
            cv2.imwrite(save_dir+f'\\DEPTH\\{i}.png', depth_image)

            # 保存 pmat 为 PNG 格式
            cv2.imwrite(save_dir+f'\\PMAT\\{i}.png', pmat)
        
            # print(color_image,depth_image,pmat)
            print(f"Frame [{i}] saved!")
            i += 1

        elif save_stream == 'Manual'and key == ord('s'):  # 按下 's' 键进行截图
            os.makedirs(save_dir+'\\RGB', exist_ok=True)
            os.makedirs(save_dir+'\\DEPTH', exist_ok=True)
            os.makedirs(save_dir+'\\PMAT', exist_ok=True)
            # 保存 color_image 为 PNG 格式
            cv2.imwrite(save_dir+f'\\RGB\\{i}.png', color_image)

            # 保存 depth_image 为 PNG 格式
            cv2.imwrite(save_dir+f'\\DEPTH\\{i}.png', depth_image)

            # 保存 pmat 为 PNG 格式
            cv2.imwrite(save_dir+f'\\PMAT\\{i}.png', pmat)
            print(f"Frame [{i}] saved!")
            i += 1

        elif save_stream == 'Auto' and motion_detection(P_frame1, P_frame2, threshold=1) and square_count(pmat)>80:  # 按下 's' 键进行截图
            os.makedirs(save_dir+'\\RGB', exist_ok=True)
            os.makedirs(save_dir+'\\DEPTH', exist_ok=True)
            os.makedirs(save_dir+'\\PMAT', exist_ok=True)
            # 保存 color_image 为 PNG 格式
            cv2.imwrite(save_dir+f'\\RGB\\{i}.png', color_image)

            # 保存 depth_image 为 PNG 格式
            cv2.imwrite(save_dir+f'\\DEPTH\\{i}.png', depth_image)

            # 保存 pmat 为 PNG 格式
            cv2.imwrite(save_dir+f'\\PMAT\\{i}.png', pmat)
            print(f"Frame [{i}] saved!")
            i += 1

        ret, marker_ids, tag_pose = CAM.detect_markers(color_image)

        # color_image = np.rot90(color_image,k = 1)

        # 在此处添加对标记物的处理逻辑
        if ret:
            for markerID in marker_ids:
                CAM.draw_axis(color_image, tag_pose[f'tag{markerID}_rvecs'], tag_pose[f'tag{markerID}_tvecs'], length = 100)
            # print("corners: ",tag_pose[f'corners'])
            # print(color_image.shape)
            cv2.aruco.drawDetectedMarkers(color_image, tag_pose[f'corners'])

        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

        # 显示图像
        cv2.imshow("RGB", color_image)
        # cv2.resizeWindow("RGB", 450, 800)
        cv2.setMouseCallback("RGB", mouse_callback, color_image)
        # cv2.imshow("grey", grey_img)
        cv2.imshow("D", depth_image)
        # cv2.resizeWindow("D", 450, 800)
        cv2.setMouseCallback("D", mouse_callback, depth_image)
        flipped_pmat = pmat[:, ::-1]
        cv2.imshow("PMAT", cv2.resize(flipped_pmat, (flipped_pmat.shape[1]*10, flipped_pmat.shape[0]*10)))
        

        
        # 计算已经过的时间
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        # 控制采样速度为每秒x帧
        delay_time = 1 / desired_fps - elapsed_time
        
        if delay_time > 0:
            time.sleep(delay_time)

        
        RGB_frame1 = color_image
        P_frame1 = pmat

    
    # 停止采集
    print(f"total time  = {total_time:.2f}s")
    print("关闭数据采集pipline...")
    CAM.stop_capture()
    PMAT.stop_capture()

    sleep(0.5)
    print("\n--------采集结束--------")
    if save_stream == "Stream":
        print(f"- 采集模式: 全程录制")
        print(f"- 视频帧率: {desired_fps} fps")
    elif save_stream == "Manual":
        print(f"- 采集模式: 手动抽帧")
    elif save_stream == "Auto":
        print(f"- 采集模式: 自动抽帧")

    print(f"- 共采集: {i}帧")
    print(f"- 总时间跨度: {total_time:.2f}s")
    print(f"- 保存路径: {save_dir}")
    # 关闭图像显示窗口
    cv2.destroyAllWindows()
    




if __name__ == "__main__":
    get_data(D455Capture, YotliveCapture, save_stream = 'Auto', save_path =  r'E:\WorkSpace\inbed-pose\data_collect\dataset\test0')

    