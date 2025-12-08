import rclpy
import DR_init
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import threading
import math

# =========================================================
# [설정] 사용자 환경에 맞게 수정하세요
# =========================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA"

# 학습된 OBB 모델 경로 (반드시 본인 경로로 수정!)
# 예: "runs/obb/train/weights/best.pt"
MODEL_PATH = "/home/rokey/ros2_ws/src/tracking/tracking/best_obb1.pt"

# --- 튜닝 파라미터 ---
LOOP_RATE = 0.1         # 명령 주기 (초)
MOVE_TIME = 5         # 로봇 이동 시간 (부드러움을 위해 주기보다 길게 설정)

# 위치 제어 게인 (속도 조절)
GAIN_XY = 0.2           # 상하좌우 반응성 (0.1 ~ 0.5)
GAIN_Z = 0.3            # 앞뒤 반응성

# [NEW] 회전 제어 게인
GAIN_W = 0.5            # 회전 반응성 (너무 크면 진동 발생)

# 안전 제한 (Safety)
MAX_STEP = 50.0         # 한 번에 이동할 최대 거리 (mm)
MAX_ROT  = 10.0         # 한 번에 회전할 최대 각도 (도, degree)
TARGET_DIST = 0.30      # 목표 거리 (미터)
CONF_THRESHOLD = 0.85    # 인식 임계값
DEADZONE = 10           # 픽셀 오차 무시 구간

# 방향 보정 (로봇이 반대로 움직이면 -1로 변경)
DIR_X = 1
DIR_Y = -1
DIR_Z = 1
DIR_RZ = 1              # [NEW] 회전 방향 보정

# 전역 변수
shared_step = None
stop_event = False

# 두산 로봇 초기화 설정
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# =========================================================
# [Thread] 로봇 제어 (Override 방식)
# =========================================================
def robot_motion_thread():
    global shared_step, stop_event
    from DSR_ROBOT2 import amovel, DR_TOOL, DR_MV_MOD_REL, DR_MV_RA_OVERRIDE
    
    print(">>> Robot Motion Thread Started (6-DOF)")
    
    while not stop_event:
        start_t = time.time()
        
        if shared_step is not None:
            # 6축 데이터 언팩 (x, y, z, rx, ry, rz)
            dx, dy, dz, drx, dry, drz = shared_step
            
            # 움직임이 미세하게라도 있을 때만 명령 전송
            if sum(abs(v) for v in [dx, dy, dz, drx, dry, drz]) > 0.001:
                try:
                    # [핵심] 6축 상대 이동 명령
                    # pos: [x, y, z, rx, ry, rz]
                    target_pos = [dx, dy, dz, drx, dry, drz]
                    
                    amovel(target_pos, 
                           time=MOVE_TIME, 
                           mod=DR_MV_MOD_REL, 
                           ref=DR_TOOL,          # 툴 기준 (카메라가 툴에 달렸다고 가정)
                           ra=DR_MV_RA_OVERRIDE) # 부드러운 덮어쓰기 모드
                    
                except Exception as e:
                    print(f"Motion Error: {e}")
        
        # 주기 유지
        elapsed = time.time() - start_t
        sleep_time = LOOP_RATE - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# =========================================================
# [Main] 비전 처리 및 제어량 계산
# =========================================================
def main(args=None):
    global shared_step, stop_event
    
    rclpy.init(args=args)
    node = rclpy.create_node("obb_tracker", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    from DSR_ROBOT2 import set_tool, set_tcp

    # 로봇 설정
    set_tool(ROBOT_TOOL)
    set_tcp(ROBOT_TCP)

    # 리얼센스 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print(">>> RealSense 시작 중...")
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # YOLO 모델 로드
    print(f">>> 모델 로드 중: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"!!! 모델 로드 실패: {e}")
        print("경로를 확인하거나 'yolo11n-obb.pt'를 사용해 보세요.")
        pipeline.stop()
        return

    # 스레드 시작
    t = threading.Thread(target=robot_motion_thread)
    t.daemon = True
    t.start()

    print(">>> OBB Tracking 시작 (Press 'q' to exit)")

    try:
        while rclpy.ok():
            start_time = time.time()

            # 1. 이미지 취득
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame: continue

            img = np.asanyarray(color_frame.get_data())
            depth_map = np.asanyarray(depth_frame.get_data())
            
            # (옵션) 보기 편하게 좌우 반전 (로봇 제어 방향 헷갈림 주의)
            # img = cv2.flip(img, 1)
            # depth_map = cv2.flip(depth_map, 1)

            # 2. YOLO OBB 추론
            results = model.track(img, persist=True, verbose=False)
            annotated_frame = results[0].plot() # 시각화
            
            # 화면 중심점
            cv2.circle(annotated_frame, (320, 240), 5, (0, 0, 255), -1)

            dist_m = 0.0
            # 제어 변수 초기화
            step_x, step_y, step_z = 0.0, 0.0, 0.0
            step_rx, step_ry, step_rz = 0.0, 0.0, 0.0

            # 3. 데이터 처리
            # results[0].obb가 존재하는지 확인
            if results[0].obb is not None and len(results[0].obb) > 0:
                
                # 가장 신뢰도 높은 객체 하나 선택
                # OBB 데이터 형식: [cx, cy, w, h, rotation_radian]
                obb_data = results[0].obb.xywhr[0].cpu().numpy()
                conf = results[0].obb.conf[0].item()

                if conf >= CONF_THRESHOLD:
                    cx, cy = int(obb_data[0]), int(obb_data[1])
                    rotation_rad = obb_data[4] # 라디안 각도 (-pi/2 ~ pi/2)
                    
                    # 중심점 연결선 그리기
                    cv2.line(annotated_frame, (320, 240), (cx, cy), (0, 255, 0), 2)

                    # 거리(Depth) 가져오기
                    if 0 <= cy < 480 and 0 <= cx < 640:
                        dist_m = depth_map[cy, cx] * depth_scale
                    else: dist_m = 0

                    if dist_m > 0:
                        # --- [A] 위치 제어 (XYZ) ---
                        err_x = cx - 320
                        err_y = cy - 240
                        err_z = dist_m - TARGET_DIST

                        if abs(err_x) < DEADZONE: err_x = 0
                        if abs(err_y) < DEADZONE: err_y = 0
                        if abs(err_z) < 0.02:     err_z = 0

                        step_x = err_x * GAIN_XY * DIR_X
                        step_y = err_y * GAIN_XY * DIR_Y
                        step_z = err_z * 1000.0 * GAIN_Z * DIR_Z

                        # --- [B] 회전 제어 (RZ - Yaw) ---
                        # 목표: 물체가 카메라에서 수직(0도)으로 보이도록 로봇을 회전
                        # Eye-in-Hand 구조에서는 물체 각도만큼 로봇을 회전시키면 됨
                        current_angle_deg = math.degrees(rotation_rad)
                        
                        # 각도 오차에 대한 게인 적용
                        # OBB 각도가 90도 단위로 튀는 현상을 막으려면 별도 로직이 필요할 수 있음
                        step_rz = current_angle_deg * GAIN_W * DIR_RZ

            # 4. 안전 제한 (Clipping)
            step_x = np.clip(step_x, -MAX_STEP, MAX_STEP)
            step_y = np.clip(step_y, -MAX_STEP, MAX_STEP)
            step_z = np.clip(step_z, -MAX_STEP, MAX_STEP)
            step_rz = np.clip(step_rz, -MAX_ROT, MAX_ROT) # 회전 제한

            # 5. 스레드로 데이터 전송 (6축)
            shared_step = [step_x, step_y, step_z, step_rx, step_ry, step_rz]

            # 6. 정보 출력
            fps = 1.0 / (time.time() - start_time)
            info_txt = f"FPS: {fps:.1f} | Z: {dist_m:.2f}m | RZ: {step_rz:.1f}"
            cv2.putText(annotated_frame, info_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("OBB 6-DOF Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event = True
                break

    except Exception as e:
        print(f"Main Error: {e}")
        stop_event = True
    finally:
        if pipeline: pipeline.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


