import rclpy
import DR_init
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

# --- 로봇 설정 상수 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA"

# --- [핵심] 방향 보정 설정 (1: 정방향, -1: 역방향) ---
# 움직임이 반대면 이 숫자를 -1로 바꾸세요.
DIR_X = 1   # 좌우 방향 (오른쪽으로 갈 때 로봇도 오른쪽으로 가면 1)
DIR_Y = 1   # 상하 방향 (아래로 갈 때 로봇도 아래로 가면 1)
DIR_Z = 1   # 전후 방향 (멀어질 때 로봇도 전진하면 1)

# --- 파라미터 ---
VELOCITY = 30
ACC = 30
TARGET_DIST = 0.40
MOVE_THRESHOLD = 0.02
MAX_STEP = 50.0
CONF_THRESHOLD = 0.7

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

def initialize_robot():
    from DSR_ROBOT2 import set_tool, set_tcp
    set_tool(ROBOT_TOOL)
    set_tcp(ROBOT_TCP)

def connect_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    intrinsics = depth_profile.get_intrinsics()
    return pipeline, align, intrinsics, profile

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("tuning_tracker", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    from DSR_ROBOT2 import posx, movel, DR_TOOL

    pipeline = None
    try:
        initialize_robot()
        pipeline, align, intrinsics, profile = connect_realsense()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        try:
            model = YOLO("/home/rokey/ros2_ws/src/tracking/tracking/best.pt")
        except:
            model = YOLO("yolo11n.pt")

        print(">>> 방향 튜닝 모드 시작")
        print(f">>> 현재 설정: X({DIR_X}), Y({DIR_Y}), Z({DIR_Z})")

        while rclpy.ok():
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
            
            target_found = False
            move_x, move_y, move_z = 0, 0, 0
            status_text = "Waiting..."

            if results[0].boxes.id is not None:
                conf = results[0].boxes.conf[0].item()
                if conf >= CONF_THRESHOLD:
                    box = results[0].boxes.xyxy.cpu().numpy()[0]
                    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                    
                    if 0 <= cy < 480 and 0 <= cx < 640:
                        dist_m = depth_map[cy, cx] * depth_scale
                    else: dist_m = 0

                    if dist_m > 0:
                        target_found = True
                        
                        # 오차 계산
                        err_x = cx - 320
                        err_y = cy - 240
                        err_z = dist_m - TARGET_DIST

                        # [방향 보정 적용] DIR 변수를 곱해서 방향 결정
                        move_x = err_x * 0.2 * DIR_X
                        move_y = err_y * 0.2 * DIR_Y
                        move_z = err_z * 1000.0 * DIR_Z

                        status_text = f"X:{move_x:.0f} Y:{move_y:.0f} Z:{move_z:.0f}"

            # 이동 실행
            if target_found and (abs(move_x)>5 or abs(move_y)>5 or abs(move_z)>10):
                move_x = np.clip(move_x, -MAX_STEP, MAX_STEP)
                move_y = np.clip(move_y, -MAX_STEP, MAX_STEP)
                move_z = np.clip(move_z, -MAX_STEP, MAX_STEP)

                delta_pos = posx([move_x, move_y, move_z, 0, 0, 0])
                movel(delta_pos, vel=VELOCITY, acc=ACC, ref=DR_TOOL)
                
                cv2.putText(annotated_frame, "MOVING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow("Tuning", annotated_frame)
                cv2.waitKey(1)
            else:
                cv2.putText(annotated_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Tuning", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if pipeline: pipeline.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()