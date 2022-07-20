# -*- coding: utf-8 -*-
# author: novauto chenyifei
# 2022.6.22

import argparse
import numpy as np
import os, sys, shutil
import mayavi.mlab as mlab
import kitti_util
import time
from tqdm import tqdm
import cv2
import glob

def parser_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='kitti',help='specify the format')  # inference
    # inference: predict result format is .bin (lidar coors)
    # kitti: predict result format is .txt (camera coors) 
    parser.add_argument('--lidar',type=str,default=r'C:\\Users\\ptrgu\\Downloads\\pc-tools-master-pc_tools\\pc-tools-master-pc_tools\\pc_tools\\ground_calib_benewake',help='the lidar file')
    # lidar filepath
    parser.add_argument('--label',type=str, default=False,help='the labels file')    ###r'C:\Users\ptrgu\Desktop\label_2'
    # label filepath
    parser.add_argument('--calib',type=str, default=r'C:\Users\ptrgu\Desktop\calib',help='the calib file')
    # calib filepath
    parser.add_argument('--predict',type=str, default=r'C:\Users\ptrgu\Downloads\detection_demo\data',help='the detection file')
    # predict filepath
    parser.add_argument('--draw_gt',type=bool, default=False, help='draw gtbox or not')
    # draw gt box
    parser.add_argument('--draw_pre',type=bool, default=True, help='draw prebox or not')
    # draw predict box
    parser.add_argument('--save',type=str, default=r'./save_demo', help='save img file')
    # save img demo path
    parser.add_argument('--vis',type=bool, default=True,help='vis or not, False:3D  True:2D image')
    # save img or vis
    parser.add_argument('--score',type=float, default=0.2,help='score thresh')
    # score thresh
    parser.add_argument('--video',type=str, default='detection_demo',help='video name')
    # video file 
    args = parser.parse_args()
    return args


def center_to_corner_box3d(boxes_center, coordinate='lidar'):
    # (N, 7) -> (N, 8, 3) #中心点坐标转换为立方体8个角坐标
    if coordinate != 'lidar':
        raise RuntimeError('error')
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]#中心坐标
        size = box[3:6]  #框的尺寸
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
			[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
			[w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
			#[0, 0, 0, 0, h, h, h, h]])

		# re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
			[np.sin(yaw), np.cos(yaw), 0.0],
			[0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
			np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    return ret


def draw_gt_boxes3d(gt_boxes3d, colors, fig, line_width=1, draw_text=True, text_scale=(0.5,0.5,0.5)):
    gt_boxes3d = center_to_corner_box3d(gt_boxes3d, coordinate='lidar') #(n,8,3)
    num = len(gt_boxes3d)

    for n in range(num):
        b = gt_boxes3d[n]
        name = 'test'
        color = colors[n]
        #mlab.text3d(b[4,0], b[4,1], b[4,2], '%s'%name, scale=text_scale, color=color, figure=fig)  #%n
        for k in range(0,4):
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

        center_top = b[4:].mean(0)
        center_front_top = b[4:6].mean(0)
        mlab.plot3d([center_top[0], center_front_top[0]], [center_top[1], center_front_top[1]], [center_top[2], center_front_top[2]], color=(0,1,0), tube_radius=None, line_width=line_width, figure=fig)
    return fig


def camera_to_lidar_box(boxes, V2C=None, R0=None, P2=None):
	# (N, 7) -> (N, 7) x,y,z,h,w,l,r
	ret = []
	for box in boxes:
		x, y, z, h, w, l, ry = box
		(x, y, z), h, w, l, rz = camera_to_lidar(
			x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -ry - np.pi / 2
		#rz = angle_in_limit(rz)
		ret.append([x, y, z, h, w, l, rz])
	return np.array(ret).reshape(-1, 7)


def inverse_rigid_trans(Tr):
	''' Inverse a rigid body transform matrix (3x4 as [R|t])
		[R'|-R't; 0|1]
	'''
	inv_Tr = np.zeros_like(Tr) # 3x4
	inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
	inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
	return inv_Tr

def camera_to_lidar(x, y, z, V2C=None,R0=None,P2=None):
	# (1, 3) -> (1, 3)
	p = np.array([x, y, z, 1])

	R0_i = np.zeros((4,4))
	R0_i[:3,:3] = R0
	R0_i[3,3] = 1
	p = np.matmul(np.linalg.inv(R0_i), p)
	p = np.matmul(inverse_rigid_trans(V2C), p)
	p = p[0:3]
	return tuple(p)

def lidar_to_camera(x, y, z, V2C=None, R0=None):
	# (1, 3) -> (1, 3)
	lidar_vel0 = np.array([x, y, z, 1])
	rect = np.matmul(V2C, lidar_vel0)
	cam_box = np.matmul(R0, rect).T
	cam_box = cam_box[0:3].reshape(-1,3)
	x, y, z = cam_box[:,0],cam_box[:,1],cam_box[:,2]
	return round(float(x),2), round(float(y),2), round(float(z),2)

def project_rect_to_image(self, pts_3d_rect):
	''' Input: 1x3 points in rect camera coord.
		Output: 1x2 points in image2 coord.
	''' 
	pts_3d_rect = self.cart2hom(pts_3d_rect)
	pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
	pts_2d[:,0] /= pts_2d[:,2]
	pts_2d[:,1] /= pts_2d[:,2]
	return pts_2d[:,0:2]

def lidar_to_camera_point(points, V2C=None, R0=None):
	# (N, 3) -> (N, 3)
	N = points.shape[0]
	points = np.hstack([points, np.ones((N, 1))]).T

	if V2C is None or R0 is None:
		print("error!")
	else:
		points = np.matmul(V2C, points)
		points = np.matmul(R0, points).T
	points = points[:, 0:3]
	return points.reshape(-1, 3)

def gene_video(save_path, video_name, video_format, fps):
    img_list = sorted(glob.glob(save_path + "/*.jpg"))
    img = cv2.imread(img_list[0])
    h, w, c = img.shape
    size = (w, h)

    video_path = os.path.join(save_path, video_name+video_format) # .mp4 --> M J P G
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('X','V','I','D'), fps, size)

    for i in tqdm(range(len(img_list))):
        items = img_list[i]
        img = cv2.imread(items)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print("****** video path:", video_path, "*********")

if __name__ == '__main__':

    args = parser_config()
    mode         = args.mode
    lidar_path   = args.lidar
    label_path   = args.label
    pre_path     = args.predict
    calib_path   = args.calib
    save_path    = args.save
    visual       = args.vis
    score_thresh = args.score
    video_name   = args.video


    if visual:
        mlab.options.offscreen = True
    else:
        mlab.options.offscreen = False

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # gene lidar detection img
    if mode == 'kitti':
        print('******** pointcloud format: .bin')
        print('******** label format: .txt')
        # for pre_file in os.listdir(pre_path):label_path
        for pre_file in os.listdir(pre_path):
            lidar_f = pre_file.replace('txt', 'bin')
            lidar_file = os.path.join(lidar_path, lidar_f)
            # gt_label_path = os.path.join(label_path, pre_file)
            pre_label_path = os.path.join(pre_path, pre_file)
            cal_path = os.path.join(calib_path, "000000.txt")
            calib = kitti_util.Calibration(cal_path)
            points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
            fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1300, 600))
            color = points[:, 3]
            color = np.clip(color+0.3, 0, 1)
            mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color, color=None, mode='point', colormap='GnBu', scale_factor=2, figure=fig)
            
            # COLOR = {'Car':(0,0,1.0),'Pedestrian':(1.0,0.0,0),'Cyclist':(0, 1.0, 0)}
            # determine by dataset classes
            # COLOR = {'Car':(1.0,0,0),'Van':(0.0,0.0,1.0),'Cyclist':(0, 1.0, 0), 'Bus':(1.0,0.0,1.0), 
            #         'Tricyclist':(0, 1.0, 1.0), 'Pedestrian':(1.0,1.0,0.0)}
            COLOR = {'Car':(1.0,0,0), 'Bus':(1.0,0.0,1.0)}
            if args.draw_gt:

                with open(gt_label_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        continue
                m = 0
                labels = []
                for line in lines:
                    line = line.strip().split(' ')
                    try:
                        cls_type = line[0]
                        x = float(line[11])
                        y = float(line[12])
                        z = float(line[13])
                        l = float(line[10])
                        w = float(line[9])
                        h = float(line[8])
                        r = -float(line[14])-1/2*np.pi+np.pi
                        x, y, z = camera_to_lidar(x, y, z, V2C=calib.V2C,R0=calib.R0,P2=calib.P)
                        z = z + 1/2*h
                        # if np.abs(x)<=52 and np.abs(y)<=24:
                        boxes_center = np.array([x,y,z, h, w, l, r], dtype=np.float32).reshape(1, -1)
                        color = (1.0,1.0,1.0)
                        fig = draw_gt_boxes3d(boxes_center, [color], fig, line_width=1.5, draw_text=True, text_scale=(0.5,0.5,0.5))
                        m += 1
                        # print('{}/{}'.format(m, len(lines)))
                    except KeyError:
                        continue

            if args.draw_pre:
                with open(pre_label_path, 'r') as p:
                    lines_2 = p.readlines()
                    if len(lines_2) == 0:
                        continue

                m = 0
                for line in lines_2:
                    line = line.strip().split(' ')
                    if float(line[15]) <= score_thresh:
                        break
                    try:
                        cls_type = line[0]
                        x = float(line[11])
                        y = float(line[12])
                        z = float(line[13])
                        l = float(line[10])
                        w = float(line[9])
                        h = float(line[8])
                        r = -float(line[14])-1/2*np.pi+np.pi
                        x, y, z = camera_to_lidar(x, y, z, V2C=calib.V2C,R0=calib.R0,P2=calib.P)
                        z = z + 1/2*h

                        # if np.abs(x)<=40 and np.abs(y)<=20:
                        boxes_center = np.array([x,y,z, h, w, l, r], dtype=np.float32).reshape(1, -1)
                        color = COLOR[cls_type]
                        fig = draw_gt_boxes3d(boxes_center, [color], fig, line_width=1.5, draw_text=True, text_scale=(0.5,0.5,0.5))
                        m += 1
                        # print('{}/{}'.format(m,len(lines_2)))
                    except KeyError:
                        continue

            mlab.view(azimuth=180, elevation=45, focalpoint=[70, 0, 0], distance=170.0, figure=fig)
            if not visual: # 3D visualization
                mlab.options.offscreen = False
                mlab.show()
                input()
            else:  # 2d image
                mlab.options.offscreen = True
                save_paths = os.path.join(save_path, pre_file.split('.')[0]+'.jpg')
                mlab.savefig(save_paths, figure=mlab.gcf())
                print("----- ", save_paths, " ----saved!")
                mlab.close(fig)

    elif mode == 'inference':
        print('******** pointcloud path:', lidar_path, ' format: .bin ********')
        print('******** detections path:', pre_path, 'format: .bin ********')
        print('******** visual score:', score_thresh,  '********')

        for pre in os.listdir(pre_path):
            raw = pre.replace('_0.bin','.bin')
            lidar_file = os.path.join(lidar_path, raw)
            pre_file = os.path.join(pre_path, pre)
            points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
            results = np.fromfile(pre_file, dtype=np.float32).reshape(-1, 9)
            if results.shape[0] == 0:
                continue

            fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1300, 600))
            color = points[:, 3]
            color = np.clip(color+0.3, 0, 1)
            mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color, color=None, mode='point', colormap='GnBu', scale_factor=1, figure=fig)
            
            # determine by dataset classes
            COLOR = {'bicycle':(1.0,0,0),'big_vehicle':(0.0,0.0,1.0),'vehicle':(0, 1.0, 0), 'pedestrian':(0.8,0,0.8), 
                    'huge_vehicle':(0.0, 1.0, 1.0)}
            classes = ['vehicle', 'bicycle', 'pedestrian']
            m = 0

            for i in range(results.shape[0]):
                if sum(results[i]) <= 1:
                    break
                if results[i][7] <= score_thresh:
                    break
                x, y, z, l, w, h, r, s, c = results[i]
                r = r + np.pi
                cls_type = classes[int(c)]
                boxes_center = np.array([x,y,z, h, w, l, r], dtype=np.float32).reshape(1, -1)
                color = COLOR[cls_type]
                fig = draw_gt_boxes3d(boxes_center, [color], fig, line_width=1.5, draw_text=True, text_scale=(0.5,0.5,0.5))
                m += 1
                # print('{}/{}'.format(m, len(results)))

            mlab.view(azimuth=180, elevation=45, focalpoint=[15, 0, 0], distance=70.0, figure=fig)

            if not visual: # 3D visualization
                mlab.options.offscreen = False
                mlab.show()
                input()
            else:  # 2d image
                mlab.options.offscreen = True
                save_paths = os.path.join(save_path, raw.split('.')[0]+'.jpg')
                mlab.savefig(save_paths, figure=mlab.gcf())
                print("----- ", save_paths, " ----saved!")
                mlab.close(fig)
    else:
        print("mode error!")


    # gene lidar detection video
    video_format = ".avi"
    fps = 10
    gene_video(save_path, video_name, video_format, fps)

    