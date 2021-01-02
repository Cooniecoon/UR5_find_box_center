#!/usr/bin/env python

import torch
import numpy as np
from numpy import random
import socket 
import cv2
import time
from math import tan, asin, pi, sqrt, acos, sin, cos


from models.experimental import attempt_load
from utils.plots import plot_one_box
from utils.general import non_max_suppression


model_path='box.pt'
def cali_L(source):
    rms = 0.219657
    fx = 625.439940
    fy = 625.439940
    cx = 320.000000
    cy = 240.000000
    k1 = 0.031886
    k2 = -0.024753
    p1 = -0.008296
    p2 = -0.006986

    hfov = 54.2
    vfov = 42.0 #deg

    # A (Intrinsic Parameters) [fc, skew*fx, cx], [0, fy, cy], [0, 0, 1]
    K = np.array([[fx, 0., cx],
                [0,  fy, cy],
                [0,   0,  1]])

    # Distortion Coefficients(kc) - 1st, 2nd
    d = np.array([k1, k2, p1, p2, 0]) # just use first two terms

    image = source
    img = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]

    # undistort
    newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)
    newimg = cv2.undistort(img, K, d, None, newcamera)
    img = cv2.resize(newimg, (640, 480), interpolation=cv2.INTER_LINEAR)

    return img

def cali_R(source):
    rms = 0.223110
    fx = 633.992978
    fy = 633.992978
    cx = 320.000000
    cy = 240.000000
    k1 = 0.032298
    k2 = -0.054036
    p1 = -0.008588
    p2 = 0.005842

    hfov = 54.2
    vfov = 42.0

    # A (Intrinsic Parameters) [fc, skew*fx, cx], [0, fy, cy], [0, 0, 1]
    K = np.array([[fx, 0., cx],
                [0,  fy, cy],
                [0,   0,  1]])

    # Distortion Coefficients(kc) - 1st, 2nd
    d = np.array([k1, k2, p1, p2, 0]) # just use first two terms

    image = source
    img = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]

    # undistort
    newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)
    newimg = cv2.undistort(img, K, d, None, newcamera)
    img = cv2.resize(newimg, (640, 480), interpolation=cv2.INTER_LINEAR)

    return img

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocessing(img):
    img = letterbox(img, new_shape=(640,640))[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to('cuda:0')
    img = img.half() #if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)
    return img

def bbox_center(bbox):
    if len(bbox):
        cx=bbox[0]+(bbox[2]-bbox[0])//2
        cy=bbox[1]+(bbox[3]-bbox[1])//2
        return [cx, cy]
        
def calculate_area(bbox):
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

def tf_world_base(bbox, X,Y,Z, side): # X, Y, Z : UR5 TCP coordinate
    cx = 320.000000
    cy = 240.000000

    if side == 'left':
        f = 625.439940
    elif side == 'right':
        f = 633.992978

    #Camer Offset from tcp
    offset_X = 30
    offset_Y = 60.8
    offset_Z = 49.15

    hfov = 54.2 #    : x
    vfov = 42.0 #deg : y

    hfov = hfov*3.1415/180
    vfov = vfov*3.1415/180

    cZ = Z - offset_Z # Camera Z coordinate

    p = bbox_center(bbox)
    px = p[0]
    py = p[1]

    Xc = cZ*(px-cx)/f
    Yc = cZ*(py-cy)/f

    if side == 'left':
        X = Xc + X - offset_X
    elif side == 'right':
        X = Xc + X + offset_X

    Y = -1 * Yc + Y - offset_Y

    return X,Y

def tf_camera_base(bbox, X,Y,Z, side):
    cx = 320.000000
    cy = 240.000000

    if side == 'left':
        f = 625.439940
    elif side == 'right':
        f = 633.992978

    #Camer Offset from tcp
    offset_X = 30
    offset_Y = 60.8
    offset_Z = 49.15

    hfov = 54.2 #    : x
    vfov = 42.0 #deg : y

    hfov = hfov*3.1415/180
    vfov = vfov*3.1415/180

    cZ = Z - offset_Z - 12.36 # Camera Z coordinate

    p = bbox
    px = p[0]
    py = p[1]

    Xc = cZ*(px-cx)/f
    Yc = cZ*(py-cy)/f

    return Xc, -Yc

def Box_Angle(bbox, Z):
    print('Final Bbox : ',bbox)
    box_X = 222
    box_Y = 90
    box_Z = 142
    alpha = 22.068

    offset_Z = 49.15

    box_cZ = Z - offset_Z - box_Z - 16

    cY = bbox[3]-bbox[1]
    cX = bbox[2]-bbox[0]

    f = 625.439940
    pX = f*box_X/box_cZ
    pY = f*box_Y/box_cZ
    pL = sqrt(pX*pX+pY*pY)
    


    beta_s = asin(cY/pL)*180/pi-alpha
    # print(beta_s)
    # beta_c = alpha + acos(cX/pL)*180/pi
    # print('\n\n')
    # print('sin : ', beta_s, '   cos : ', beta_c)
    # print('\n\n')
    return beta_s
    

def answer():
    answer = 'answer'
    server_socket.sendall(answer.encode())
    # print('answer')

### socket setting ###
HOST = '192.168.12.102'  
PORT = 6666
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.connect((HOST, PORT))

print('connected')
model = attempt_load(model_path, map_location='cuda')
model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS
model.half()
names = model.module.names if hasattr(model, 'module') else model.names
# print(names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


cap_L=cv2.VideoCapture(0)
# cap_R=cv2.VideoCapture(2)

cap_L.set(3,640)
cap_L.set(4,480)
# cap_R.set(3,640)
# cap_R.set(4,480)
### UR connect ###

err = 0.3 # mm
Iter = 10
dX = 100
dY = 100



count = 0
i=0
while True:
    data = server_socket.recv(1024)
    #print('Received', repr(data.decode()))
    if data.decode() == "give me position" :
        answer()

        while True:
            # print('start')
            data = server_socket.recv(1024)
            current_pose = data.decode()
            if current_pose:
                answer()
            # print(current_pose)
            current_pose = current_pose.split(',')
            

            X = float(current_pose[0])*1000
            Y = float(current_pose[1])*1000
            Z = float(current_pose[2])*1000

            _,img_L=cap_L.read()
            img_L=cali_L(img_L)
            im0_L=cali_L(img_L)
            img_L=preprocessing(img_L)
            # Inference
            prediction_L = model(img_L)[0]
            prediction_L = non_max_suppression(prediction_L)
            xyxy_L=prediction_L[0].cpu().numpy()

            # print(xyxy_L)
            if xyxy_L.shape[0]>0:
                i+=1
                x1_L=int(xyxy_L[0][0])
                y1_L=int(xyxy_L[0][1])
                x2_L=int(xyxy_L[0][2])
                y2_L=int(xyxy_L[0][3])

                # print('\n xyxy_L : ', x1_L, y1_L, x2_L, y2_L)
                cv2.rectangle(im0_L, (x1_L,y1_L), (x2_L,y2_L), (0,0,255), 3, lineType=cv2.LINE_AA)
                center_L=bbox_center([x1_L,y1_L,x2_L,y2_L])
                cv2.circle(im0_L,(center_L[0],center_L[1]),5,(255,0,0),-1)
                cv2.circle(im0_L,(320,240),7,(0,0,255),)

                # print('PIXEL distance ',', x : ',center_L[0]-320,', y : ', center_L[1]-240)   

                

                dX, dY = tf_camera_base(center_L, X,Y,Z, 'left')    
                # print(dX, dY)          
                print(' i : ',i)
                if (abs(dX) < err and abs(dY) < err) or (i>Iter and abs(dX) < 1 and abs(dY) < 1) :
                    count+=1
                    Rz = Box_Angle([x1_L,y1_L,x2_L,y2_L],Z)
                    area=calculate_area([x1_L,y1_L,x2_L,y2_L])
                    cv2.line(im0_L,(int(320-500*sin((Rz-90)*pi/180)),int(240-500*cos((Rz-90)*pi/180))),(int(320+500*sin((Rz-90)*pi/180)),int(240+500*cos((Rz-90)*pi/180))),(255,0,255),2,cv2.LINE_AA)
                    name='box_image/line_box'+str(Rz)+'.jpg'
                    print('Area : ',area, '\nRz Angle : ',Rz,'\nbbox : ',[x1_L,y1_L,x2_L,y2_L])
                    cv2.imwrite(name,im0_L)
                    while True:
                        next_pose = str(X+dX) + ',' + str(Y+dY) + ','+ str(Rz) + ',' + 'ok'
                        # print('final_pose : ',next_pose)
                        server_socket.sendall(next_pose.encode())
                        data = server_socket.recv(1024)
                        if data.decode() == 'answer':
                            count+=1
                            break
                    break

                while True:
                    next_pose = str(X+dX) + ',' + str(Y+dY) +','+str(0)+ ',' + 'no' 
                    print('next_pose : ', next_pose)
                    server_socket.sendall(next_pose.encode())
                    data = server_socket.recv(1024)

                    if data.decode() == 'answer':
                            break
            
            name='pt_image/box'+str(i)+'.jpg'
            cv2.imshow('l',im0_L)
            cv2.imwrite(name,im0_L)
            cv2.waitKey(10)
            
            # if cv2.waitKey(10)==ord('c'):
            #     cap_L.release()
            #     cv2.destroyAllWindows()
            #     break
        


server_socket.close()