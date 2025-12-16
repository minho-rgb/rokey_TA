import rclpy
from rclpy.node import Node
import DR_init
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import threading
import math
from sensor_msgs.msg import JointState 

# =========================================================
# [설정] 환경에 맞게 수정
# =========================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "ToolWeight"
ROBOT_TCP = "GripperDA_mh"

MODEL_PATH = "/home/rokey/ros2_ws/src/tracking/tracking/best_obb_cutout.pt"

# --- [속도 조절] 매우 부드럽게 설정 ---
LOOP_RATE = 0.08
MOVE_TIME = 1 

# 게인 값을 낮춰서 로봇이 천천히 반응하도록 함
GAIN_PAN = 0.05       
GAIN_LINEAR = 0.1     
GAIN_DIST   = 0.15    

TARGET_DIST = 0.40      
DEADZONE_PIXEL = 15     
DEADZONE_DIST  = 0.02

DIR_PAN   = -1
DIR_UP    = 1
DIR_FRONT = 1

# [안전 기준]
LIMIT_J3_BUFFER = 10.0
LIMIT_J5_BUFFER = 10.0

MAX_ROT_STEP = 10.0
MAX_LIN_STEP = 80.0

# 전역 변수
shared_step = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
step_lock = threading.Lock()
stop_event = False
real_joint_deg = None 

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# =========================================================
# [Callback] JointState 구독
# =========================================================
def joint_callback(msg):
    global real_joint_deg
    try:
        if len(msg.position) >= 6:
            # 라디안 -> 디그리 변환
            deg_vals = [math.degrees(rad) for rad in msg.position[:6]]
            
            # [핵심 수정 1] 0.0 데이터 필터링
            # 모든 관절이 0.0인 경우는 실제 상황에서 거의 없음 (초기화 데이터일 확률 99%)
            # 따라서 합계가 0.1보다 클 때만 "유효한 데이터"로 인정하고 업데이트
            if sum(abs(v) for v in deg_vals) > 0.1:
                real_joint_deg = deg_vals
                
    except Exception:
        pass

# =========================================================
# [Thread] 로봇 모션 제어
# =========================================================
def robot_motion_thread():
    global shared_step, stop_event
    from DSR_ROBOT2 import amovel, DR_TOOL, DR_MV_MOD_REL, DR_MV_RA_OVERRIDE
    
    print(">>> Robot Motion Thread Started")
    
    while not stop_event:
        start_t = time.time()
        try:
            with step_lock:
                current_step = list(shared_step)
            
            if sum(abs(v) for v in current_step) > 0.001:
                amovel(current_step, 
                       time=MOVE_TIME, 
                       mod=DR_MV_MOD_REL, 
                       ref=DR_TOOL, 
                       ra=DR_MV_RA_OVERRIDE)
        except Exception:
            time.sleep(0.05)
        
        elapsed = time.time() - start_t
        if LOOP_RATE - elapsed > 0:
            time.sleep(LOOP_RATE - elapsed)

# =========================================================
# [Main] 비전 처리 & 안전 로직
# =========================================================
def main(args=None):
    global shared_step, stop_event, real_joint_deg
    
    rclpy.init(args=args)
    node = rclpy.create_node("singularity_tracker", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    topic_name = f'/{ROBOT_ID}/joint_states'
    node.create_subscription(JointState, topic_name, joint_callback, 10)
    print(f">>> Listening to {topic_name}")

    from DSR_ROBOT2 import set_tool, set_tcp
    try:
        set_tool(ROBOT_TOOL)
        set_tcp(ROBOT_TCP)
    except:
        pass

    pipeline = rs.pipeline()
    config = rs.config()
    try:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    except Exception as e:
        print(f"RS Error: {e}")
        return

    try:
        model = YOLO(MODEL_PATH)
    except:
        return

    t = threading.Thread(target=robot_motion_thread)
    t.daemon = True
    t.start()
    
    print(">>> 트래킹 시작 (Press 'q' to exit)")

    # [핵심 수정 2] 화면 표시용 변수를 루프 밖에서 초기화 (이전 값 유지용)
    curr_j_display = [0.0]*6 

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame: continue

            img = np.asanyarray(color_frame.get_data())
            depth_map = np.asanyarray(depth_frame.get_data())

            results = model.track(img, persist=True, verbose=False)
            annotated_frame = results[0].plot()
            cv2.circle(annotated_frame, (320, 240), 10, (0, 255, 255), 2)

            step_x, step_z, step_ry = 0.0, 0.0, 0.0
            dist_m = 0.0
            warning_msg = ""
            
            # --- Vision Logic ---
            target_found = False
            cx, cy = 0, 0
            
            if hasattr(results[0], 'obb') and results[0].obb is not None and len(results[0].obb) > 0:
                best_idx = results[0].obb.conf.argmax()
                obb_data = results[0].obb.xywhr[best_idx].cpu().numpy()
                conf = results[0].obb.conf[best_idx].item()
                if conf >= 0.6:
                    cx, cy = int(obb_data[0]), int(obb_data[1])
                    target_found = True
            elif hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                best_idx = results[0].boxes.conf.argmax()
                box_data = results[0].boxes.xywh[best_idx].cpu().numpy()
                conf = results[0].boxes.conf[best_idx].item()
                if conf >= 0.6:
                    cx, cy = int(box_data[0]), int(box_data[1])
                    target_found = True

            if target_found:
                cv2.line(annotated_frame, (320, 240), (cx, cy), (0, 0, 255), 2)
                if 0 <= cy < 480 and 0 <= cx < 640:
                    dist_m = depth_map[cy, cx] * depth_scale

                if dist_m > 0:
                    err_x = cx - 320
                    err_y = cy - 240
                    err_dist = dist_m - TARGET_DIST

                    if abs(err_x) < DEADZONE_PIXEL: err_x = 0
                    if abs(err_y) < DEADZONE_PIXEL: err_y = 0
                    if abs(err_dist) < DEADZONE_DIST: err_dist = 0

                    step_ry = err_x * GAIN_PAN * DIR_PAN          
                    step_x = -err_y * GAIN_LINEAR * DIR_UP        
                    step_z = err_dist * 1000.0 * GAIN_DIST * DIR_FRONT 

            # ---------------------------------------------------------
            # [Safety Logic] 깜빡임 방지 로직 적용
            # ---------------------------------------------------------
            
            # real_joint_deg가 None이 아닐 때만 갱신 (joint_callback에서 이미 0값은 걸러짐)
            if real_joint_deg is not None:
                curr_j_display = real_joint_deg # 최신 유효 값으로 갱신
                
                j3 = curr_j_display[2]
                j5 = curr_j_display[4]

                # [A] Elbow Safety
                if abs(j3) < LIMIT_J3_BUFFER:
                    warning_msg = f"WARNING: Elbow Singularity! (J3={j3:.1f})"
                    if step_z > 0: step_z = 0 
                
                # [B] Wrist Safety
                elif abs(j5) < LIMIT_J5_BUFFER:
                    warning_msg = f"WARNING: Wrist Singularity! (J5={j5:.1f})"
                    step_ry = 0
                    step_x *= 0.1
                    step_z *= 0.1

            # Motion Update
            step_x = float(np.clip(step_x, -MAX_LIN_STEP, MAX_LIN_STEP))
            step_z = float(np.clip(step_z, -MAX_LIN_STEP, MAX_LIN_STEP))
            step_ry = float(np.clip(step_ry, -MAX_ROT_STEP, MAX_ROT_STEP))

            with step_lock:
                shared_step = [step_x, 0.0, step_z, 0.0, step_ry, 0.0]

            # Display (초록색 유지)
            # 데이터가 한 번이라도 들어왔으면 초록색, 아니면 회색
            txt_color = (0, 255, 0) if real_joint_deg is not None else (200, 200, 200)
            
            j3_val = curr_j_display[2]
            j5_val = curr_j_display[4]
            
            info_txt = f"J3:{j3_val:.1f} | J5:{j5_val:.1f}"
            
            cv2.putText(annotated_frame, info_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)
            
            dist_txt = f"Dist: {dist_m:.2f}m"
            cv2.putText(annotated_frame, dist_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if warning_msg:
                cv2.putText(annotated_frame, warning_msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(annotated_frame, (0,0), (640,480), (0,0,255), 5)

            cv2.imshow("Singularity Safe Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event = True
                break

    except Exception as e:
        print(f"Main Loop Error: {e}")
        stop_event = True
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        t.join()

if __name__ == "__main__":
    main()


