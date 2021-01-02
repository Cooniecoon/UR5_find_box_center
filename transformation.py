from math import sqrt, asin, pi, acos, atan

def Box_Angle(bbox, Z):
    box_X = 222
    box_Y = 90
    box_Z = 142
    alpha = 22.068

    offset_Z = 49.15
    
    box_cZ = Z - offset_Z-box_Z -16 -16


    cY = bbox[3]-bbox[1]
    cX = bbox[2]-bbox[0]
    f = 625.439940
    pX = f*box_X/box_cZ
    pY = f*box_Y/box_cZ

    pL = sqrt(pX*pX+pY*pY)
    print(pX, pY, pL,cX,cY)

    beta_s = asin(cY/pL)*180/pi-alpha
    beta_c = alpha + acos(cX/pL)*180/pi
    print('cos : ',beta_c,'     sin : ', beta_s)



Box_Angle([78, 141, 562, 341],500) # 0deg
#Box_Angle([64, 95, 578, 388],500) # 12deg 