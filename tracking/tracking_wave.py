import rclpy
import DR_init
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import threading

# --- 로봇 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA"

# --- 튜닝 파라미터 (핵심) ---
LOOP_RATE = 0.1        # 명령 전송 주기 (0.1초 = 10Hz)
MOVE_TIME = 3       # [중요] 로봇 이동 시간 (주기보다 훨씬 길게 설정해야 부드러움)
                       # 0.1초마다 명령을 주지만, 로봇에겐 "0.5초 동안 천천히 가"라고 함.
                       # 그리고 0.1초 뒤에 새로운 명령으로 덮어씌움 (Override).

GAIN_XY = 0.2          # 반응성 (속도)
GAIN_Z = 0.3
MAX_STEP = 50.0       # 최대 이동 범위
TARGET_DIST = 0.30
CONF_THRESHOLD = 0.8
DEADZONE = 20

# 방향 보정
DIR_X = 1
DIR_Y = -1
DIR_Z = 1

shared_step = None
stop_event = False

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# ---------------------------------------------------------
# [Robot Thread] amovel을 이용한 Override 제어
# ---------------------------------------------------------
def robot_motion_thread():
    global shared_step, stop_event
    # amovel과 필요한 상수 임포트
    from DSR_ROBOT2 import posx, amovel, DR_TOOL, DR_MV_MOD_REL, DR_MV_RA_OVERRIDE
    
    print(">>> Async Movel (Override) Thread Started")
    
    while not stop_event:
        start_t = time.time()
        
        if shared_step is not None:
            dx, dy, dz = shared_step
            
            if abs(dx) > 1.0 or abs(dy) > 1.0 or abs(dz) > 1.0:
                try:
                    # --- [핵심] amovel 사용 ---
                    # 1. pos: 이동할 거리 [dx, dy, dz, ...]
                    # 2. time=MOVE_TIME: 지정된 시간 동안 이동 (속도 벡터 역할)
                    # 3. mod=DR_MV_MOD_REL: 상대 좌표 이동
                    # 4. ra=DR_MV_RA_OVERRIDE: 이전 명령을 취소하고 즉시 새 명령 실행 (부드러움의 핵심!)
                    
                    target_pos = [dx, dy, dz, 0, 0, 0]
                    
                    amovel(target_pos, 
                           time=MOVE_TIME, 
                           mod=DR_MV_MOD_REL, 
                           ref=DR_TOOL,
                           ra=DR_MV_RA_OVERRIDE) # 덮어쓰기 모드
                    
                except Exception as e:
                    print(f"Motion Error: {e}")
        
        # 주기 관리
        elapsed = time.time() - start_t
        sleep_time = LOOP_RATE - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# ---------------------------------------------------------
# [Main Thread] (이전과 동일)
# ---------------------------------------------------------
def main(args=None):
    global shared_step, stop_event
    
    rclpy.init(args=args)
    node = rclpy.create_node("amovel_tracker", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    from DSR_ROBOT2 import set_tool, set_tcp

    set_tool(ROBOT_TOOL)
    set_tcp(ROBOT_TCP)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    try:
        model = YOLO("/home/rokey/ros2_ws/src/tracking/tracking/best.pt")
    except:
        model = YOLO("yolo11n.pt")

    t = threading.Thread(target=robot_motion_thread)
    t.daemon = True
    t.start()

    print(">>> amovel(Override) 추적 시작")

    try:
        while rclpy.ok():
            start_time = time.time()

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame: continue

            img = np.asanyarray(color_frame.get_data())
            depth_map = np.asanyarray(depth_frame.get_data())
            img = cv2.flip(img, 1)
            depth_map = cv2.flip(depth_map, 1)

            results = model.track(img, persist=True, verbose=False)
            annotated_frame = results[0].plot()
            
            cv2.circle(annotated_frame, (320, 240), 5, (255, 0, 0), -1)

            step_x, step_y, step_z = 0.0, 0.0, 0.0
            
            if results[0].boxes.id is not None:
                conf = results[0].boxes.conf[0].item()
                if conf >= CONF_THRESHOLD:
                    box = results[0].boxes.xyxy.cpu().numpy()[0]
                    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                    cv2.line(annotated_frame, (320, 240), (cx, cy), (0, 255, 0), 2)

                    if 0 <= cy < 480 and 0 <= cx < 640:
                        dist_m = depth_map[cy, cx] * depth_scale
                    else: dist_m = 0

                    if dist_m > 0:
                        err_x = cx - 320
                        err_y = cy - 240
                        err_z = dist_m - TARGET_DIST

                        if abs(err_x) < DEADZONE: err_x = 0
                        if abs(err_y) < DEADZONE: err_y = 0
                        if abs(err_z) < 0.02:     err_z = 0

                        # time 기반 속도 제어이므로 Gain을 좀 더 크게 줘도 됨
                        step_x = err_x * GAIN_XY * DIR_X
                        step_y = err_y * GAIN_XY * DIR_Y
                        step_z = err_z * 1000.0 * GAIN_Z * DIR_Z

            step_x = np.clip(step_x, -MAX_STEP, MAX_STEP)
            step_y = np.clip(step_y, -MAX_STEP, MAX_STEP)
            step_z = np.clip(step_z, -MAX_STEP, MAX_STEP)

            shared_step = [step_x, step_y, step_z]

            fps = 1.0 / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("amovel Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event = True
                break

    except Exception as e:
        print(f"Error: {e}")
        stop_event = True
    finally:
        if pipeline: pipeline.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()