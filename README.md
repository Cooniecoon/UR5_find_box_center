# UR5_find_box_center
> *최종 시연 영상 :* [![Youtube Badge](https://img.shields.io/badge/Youtube-ff0000?style=flat-square&logo=youtube&link=https://youtu.be/VJ8jvAEGjhM)](https://youtu.be/VJ8jvAEGjhM)
## 2020년 3학년 2학기 협동로봇설계 프로젝트
<div align=center>
<img width="100%" src="./img/flow.png"/>
</div>
### 미션 : 비타500 병을 옮겨서 비타500 박스에 넣기 (포장) 
1. yolov5를 이용하여 비타500박스 detection
2. 비타500박스의 Boundary Box의 중심이 실제 비타500박스의 중심에 오도록 UR5 이동 (오차 보정)
3. 중심점 픽셀좌표 -> 월드좌표 변환
4. UR5 목표위치로 이동

---

#### 1. 박스 중심점 추출
<div align=left>
<img width="50%" src="./img/find_center.gif"/>
</div>

#### 2. 박스 중심점 일치
<div align=left>
<img width="50%" src="./img/box59.jpg"/>
</div>

#### 3. 기울기 추출
<div align=left>
<img width="100%" src="./img/angle_2.jpg"/>
</div>


> *기울기 추출 데모 영상 :* [![Youtube Badge](https://img.shields.io/badge/Youtube-ff0000?style=flat-square&logo=youtube&link=https://www.youtube.com/watch?v=etjk1oNPniw)](https://www.youtube.com/watch?v=etjk1oNPniw)

---
### Gripper
<div align=center>
<img width="50%" src="./img/gripper_1.png"/>
</div>

<div align=left>
<img width="100%" src="./img/gripper_2.png"/>
</div>

<div align=left>
<img width="100%" src="./img/gripper_3.png"/>
</div>

