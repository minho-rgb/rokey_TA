import rclpy
import DR_init
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
from collections import deque
import threading

# --- 로봇 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA"

# --- 파라미터 ---
VELOCITY = 40         
ACC = 40              
TARGET_DIST = 0.30    
CONF_THRESHOLD = 0.6  
SMOOTH_SIZE = 5       
DEADZONE_XY = 40      
DEADZONE_Z = 0.03     
GAIN = 0.3            

# 방향 설정 (1 or -1)
DIR_X = 1
DIR_Y = -1
DIR_Z = 1

# 전역 변수
shared_cmd = None     
new_cmd_available = False 
stop_event = False    

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# ---------------------------------------------------------
# [스레드] 로봇 제어 (변경 없음)
# ---------------------------------------------------------
def robot_control_thread():
    global shared_cmd, new_cmd_available, stop_event
    from DSR_ROBOT2 import posx, movel, DR_TOOL
    
    while not stop_event:
        if new_cmd_available and shared_cmd is not None:
            dx, dy, dz = shared_cmd
            new_cmd_available = False 
            try:
                delta_pos = posx([dx, dy, dz, 0, 0, 0])
                movel(delta_pos, vel=VELOCITY, acc=ACC, ref=DR_TOOL)
            except Exception as e:
                print(f"Move Error: {e}")
        else:
            time.sleep(0.01)

# ---------------------------------------------------------
# 메인 함수 (시각화 개선됨)
# ---------------------------------------------------------
def main(args=None):
    global shared_cmd, new_cmd_available, stop_event
    
    rclpy.init(args=args)
    node = rclpy.create_node("visual_tracker", namespace=ROBOT_ID)
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

    history_x = deque(maxlen=SMOOTH_SIZE)
    history_y = deque(maxlen=SMOOTH_SIZE)
    history_z = deque(maxlen=SMOOTH_SIZE)

    t = threading.Thread(target=robot_control_thread)
    t.daemon = True
    t.start()

    print(">>> 시각화 개선된 추적 시작")

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

            # Flip (거울 모드)
            img = cv2.flip(img, 1)
            depth_map = cv2.flip(depth_map, 1)

            results = model.track(img, persist=True, verbose=False)
            annotated_frame = results[0].plot()

            # --- [시각화 1] 화면 중앙점 (목표) 그리기 ---
            # 파란색 점 (Blue)
            cv2.circle(annotated_frame, (320, 240), 5, (255, 0, 0), -1) 
            
            cmd_x, cmd_y, cmd_z = 0, 0, 0
            
            if results[0].boxes.id is not None:
                conf = results[0].boxes.conf[0].item()
                if conf >= CONF_THRESHOLD:
                    box = results[0].boxes.xyxy.cpu().numpy()[0]
                    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                    
                    # --- [시각화 2] 인식된 내 손 위치 그리기 ---
                    # 빨간색 점 (Red)
                    cv2.circle(annotated_frame, (cx, cy), 8, (0, 0, 255), -1)
                    
                    # --- [시각화 3] 오차 선 그리기 (Target <-> Hand) ---
                    # 초록색 선
                    cv2.line(annotated_frame, (320, 240), (cx, cy), (0, 255, 0), 2)

                    # 거리 측정
                    if 0 <= cy < 480 and 0 <= cx < 640:
                        dist_m = depth_map[cy, cx] * depth_scale
                    else: dist_m = 0

                    if dist_m > 0:
                        history_x.append(cx)
                        history_y.append(cy)
                        history_z.append(dist_m)

                        if len(history_x) == SMOOTH_SIZE:
                            avg_cx = sum(history_x) / SMOOTH_SIZE
                            avg_cy = sum(history_y) / SMOOTH_SIZE
                            avg_dist = sum(history_z) / SMOOTH_SIZE

                            err_x = avg_cx - 320
                            err_y = avg_cy - 240
                            err_z = avg_dist - TARGET_DIST

                            # 데드존 체크 & 명령 생성
                            if abs(err_x) > DEADZONE_XY: cmd_x = err_x * 0.2 * DIR_X * GAIN
                            if abs(err_y) > DEADZONE_XY: cmd_y = err_y * 0.2 * DIR_Y * GAIN
                            if abs(err_z) > DEADZONE_Z:  cmd_z = err_z * 1000.0 * DIR_Z * GAIN

            # 명령 전달
            if abs(cmd_x) > 0 or abs(cmd_y) > 0 or abs(cmd_z) > 0:
                cmd_x = np.clip(cmd_x, -60, 60)
                cmd_y = np.clip(cmd_y, -60, 60)
                cmd_z = np.clip(cmd_z, -60, 60)
                
                shared_cmd = [cmd_x, cmd_y, cmd_z]
                new_cmd_available = True
                
                # 움직이는 중임을 표시 (화면 테두리 깜빡임 효과 or 텍스트)
                cv2.putText(annotated_frame, "ROBOT MOVING", (320-50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # FPS 표시
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Visual Tracking", annotated_frame)
            
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