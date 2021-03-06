# UR5_find_box_center

<div align=center>
<img width="100%" src="./image/final_demo.gif"/>
</div>

> *최종 시연 영상 :* [![Youtube Badge](https://img.shields.io/badge/Youtube-ff0000?style=flat-square&logo=youtube&link=https://youtu.be/VJ8jvAEGjhM)](https://youtu.be/VJ8jvAEGjhM)

## 2020년 3학년 2학기 협동로봇설계 프로젝트

<div align=center>
<img width="100%" src="./image/flow.png"/>
</div>

### 미션 : 비타500 병을 옮겨서 비타500 박스에 넣기 (포장) 
1. yolov5를 이용하여 비타500박스 detection
2. 비타500박스의 Boundary Box의 중심이 실제 비타500박스의 중심에 오도록 UR5 이동 (오차 보정)
3. 중심점 픽셀좌표 -> 월드좌표 변환
4. UR5 목표위치로 이동

---

#### 1. 박스 중심점 추출
- 피드백 루프 알고리즘 적용
<div align=left>
<img width="100%" src="./image/UR5_loop.png"/>
</div>
<div align=left>
<img width="60%" src="./image/find_center.gif"/>
</div>

#### 2. 박스 중심점 일치
<div align=left>
<img width="60%" src="./image/box59.jpg"/>
</div>


#### 3. 기울기 추출
<div align=center>
<img width="100%" src="./image/find_angle.gif"/>
</div>

<div align=left>
<img width="100%" src="./image/angle_2.jpg"/>
</div>


> *기울기 추출 데모 영상 :* [![Youtube Badge](https://img.shields.io/badge/Youtube-ff0000?style=flat-square&logo=youtube&link=https://www.youtube.com/watch?v=etjk1oNPniw)](https://www.youtube.com/watch?v=etjk1oNPniw)

---
### Gripper
<div align=center>
<img width="70%" src="./image/gripper_1.png"/>
</div>

<div align=center>
<img width="100%" src="./image/gripper_2.png"/>
</div>

<div align=center>
<img width="100%" src="./image/gripper_3.png"/>
</div>

---

# TEAM
- [정석훈](https://github.com/Cooniecoon)
- [김민우](https://github.com/KKminwoo)
- [송지유](https://github.com/sjyet96)
- [최승범](https://github.com/choisb818)