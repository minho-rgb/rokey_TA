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
# [설정] 환경에 맞게 수정
# =========================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "ToolWeight"
ROBOT_TCP = "GripperDA_mh"

# 모델 경로
MODEL_PATH = "/home/rokey/ros2_ws/src/tracking/tracking/best_obb_cutout.pt"

# --- 속도 및 반응성 ---
LOOP_RATE = 0.1
MOVE_TIME = 2

GAIN_PAN = 0.15
GAIN_LINEAR = 0.4
GAIN_DIST   = 0.6

# 목표 설정
TARGET_DIST = 0.40      
DEADZONE_PIXEL = 15     
DEADZONE_DIST  = 0.02

# 방향 보정
DIR_PAN   = -1
DIR_UP    = 1
DIR_FRONT = 1

# =========================================================
# [특이점(Singularity) 안전 기준 설정] - 이미지 기준
# =========================================================
# 1. Elbow Singularity: 3축이 0도 근처면 위험
LIMIT_J3_BUFFER = 10.0  # 3축 각도가 ±10도 이내면 정지

# 2. Wrist Singularity: 5축이 0도 근처면 위험
LIMIT_J5_BUFFER = 10.0  # 5축 각도가 ±10도 이내면 정지

# 3. Shoulder Singularity: 로봇 베이스 중심(0,0) 근처 위험
LIMIT_RADIUS_MIN = 200.0 # 베이스로부터 반경 200mm 이내 진입 금지

# 안전 제한 (Max Step)
MAX_ROT_STEP = 10.0
MAX_LIN_STEP = 80.0

shared_step = None
stop_event = False

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# =========================================================
# [Thread] 로봇 제어
# =========================================================
def robot_motion_thread():
    global shared_step, stop_event
    from DSR_ROBOT2 import amovel, DR_TOOL, DR_MV_MOD_REL, DR_MV_RA_OVERRIDE
    
    print(">>> Robot Motion Thread Started")
    while not stop_event:
        start_t = time.time()
        if shared_step is not None:
            dx, dy, dz, drx, dry, drz = shared_step
            
            # 이동 명령이 있을 때만 전송
            if sum(abs(v) for v in [dx, dy, dz, drx, dry, drz]) > 0.001:
                try:
                    amovel([dx, dy, dz, drx, dry, drz], 
                           time=MOVE_TIME, 
                           mod=DR_MV_MOD_REL, 
                           ref=DR_TOOL, 
                           ra=DR_MV_RA_OVERRIDE)
                except Exception:
                    pass
        
        elapsed = time.time() - start_t
        if LOOP_RATE - elapsed > 0:
            time.sleep(LOOP_RATE - elapsed)

# =========================================================
# [Main] 비전 및 제어 로직
# =========================================================
def main(args=None):
    global shared_step, stop_event
    
    rclpy.init(args=args)
    node = rclpy.create_node("singularity_tracker", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    # [중요] 관절 각도(posj)와 좌표(posx)를 읽기 위한 함수 임포트
    from DSR_ROBOT2 import set_tool, set_tcp, get_current_posj, get_current_posx

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
        model = YOLO(MODEL_PATH)
    except:
        return

    t = threading.Thread(target=robot_motion_thread)
    t.daemon = True
    t.start()
    
    print(">>> 특이점(Singularity) 방지 추적 시작")

    try:
        while rclpy.ok():
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

            # 초기화
            step_x, step_y, step_z = 0.0, 0.0, 0.0
            step_rx, step_ry, step_rz = 0.0, 0.0, 0.0
            warning_msg = "" # 경고 메시지 저장용

            # ---------------------------------------------------------
            # [1] Vision Processing (Max Confidence)
            # ---------------------------------------------------------
            if results[0].obb is not None and len(results[0].obb) > 0:
                best_idx = results[0].obb.conf.argmax()
                obb_data = results[0].obb.xywhr[best_idx].cpu().numpy()
                conf = results[0].obb.conf[best_idx].item()

                if conf >= 0.85:
                    cx, cy = int(obb_data[0]), int(obb_data[1])
                    cv2.line(annotated_frame, (320, 240), (cx, cy), (0, 0, 255), 2)

                    dist_m = 0
                    if 0 <= cy < 480 and 0 <= cx < 640:
                        dist_m = depth_map[cy, cx] * depth_scale

                    if dist_m > 0:
                        err_x = cx - 320
                        err_y = cy - 240
                        err_dist = dist_m - TARGET_DIST

                        if abs(err_x) < DEADZONE_PIXEL: err_x = 0
                        if abs(err_y) < DEADZONE_PIXEL: err_y = 0
                        if abs(err_dist) < DEADZONE_DIST: err_dist = 0

                        # [드론 모드 매핑]
                        step_ry = err_x * GAIN_PAN * DIR_PAN          
                        step_x = -err_y * GAIN_LINEAR * DIR_UP        
                        step_z = err_dist * 1000.0 * GAIN_DIST * DIR_FRONT 

            # ---------------------------------------------------------
            # [2] 특이점(Singularity) 감지 및 방지 로직 (핵심)
            # ---------------------------------------------------------
            curr_j = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
            curr_radius = 0.0
            try:
                # 1. 현재 관절 각도 (Joint) 읽기: J1~J6
                curr_j, _ = get_current_posj() 
                j3 = curr_j[2] # 3축 (Elbow)
                j5 = curr_j[4] # 5축 (Wrist)

                # 2. 현재 공간 좌표 (Cartesian) 읽기: X, Y, Z...
                curr_p, _ = get_current_posx()
                curr_x, curr_y = curr_p[0], curr_p[1]
                
                # 베이스로부터의 거리 (Radius) 계산
                curr_radius = math.sqrt(curr_x**2 + curr_y**2)

                # --- [조건 A] Elbow Singularity (3축 0도 근접) ---
                if abs(j3) < LIMIT_J3_BUFFER:
                    warning_msg = f"WARNING: Elbow Singularity! (J3={j3:.1f})"
                    # 3축이 펴져서 문제라면 더 펴지는 방향(보통 Z 전진)을 막아야 함
                    # 안전을 위해 모든 전진 이동 차단
                    if step_z > 0: step_z = 0 

                # --- [조건 B] Wrist Singularity (5축 0도 근접) ---
                elif abs(j5) < LIMIT_J5_BUFFER:
                    warning_msg = f"WARNING: Wrist Singularity! (J5={j5:.1f})"
                    # 5축이 0도면 J4, J6가 겹침 -> 회전 제어 불능
                    # 더 이상 회전하거나 복합 이동하지 못하게 감속
                    step_ry = 0
                    step_x *= 0.1 # 속도 대폭 감소
                    step_z *= 0.1

                # --- [조건 C] Shoulder Singularity (베이스 중심 근접) ---
                elif curr_radius < LIMIT_RADIUS_MIN:
                    warning_msg = f"WARNING: Shoulder Singularity! (R={curr_radius:.0f})"
                    # 로봇 몸쪽으로 들어오는 동작(후진) 차단
                    # 좌표계에 따라 다르지만, 보통 Radius가 줄어드는 방향을 막아야 함
                    # 간단히: 안전을 위해 정지
                    step_x, step_y, step_z = 0, 0, 0

            except Exception:
                pass

            # 안전 Clipping
            step_x = np.clip(step_x, -MAX_LIN_STEP, MAX_LIN_STEP)
            step_z = np.clip(step_z, -MAX_LIN_STEP, MAX_LIN_STEP)
            step_ry = np.clip(step_ry, -MAX_ROT_STEP, MAX_ROT_STEP)

            # 전송
            shared_step = [step_x, step_y, step_z, step_rx, step_ry, step_rz]

            # 상태 표시 (경고 메시지가 있으면 빨간색으로 표시)
            if warning_msg:
                cv2.putText(annotated_frame, warning_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 경고 시 화면 테두리 빨간색
                cv2.rectangle(annotated_frame, (0,0), (640,480), (0,0,255), 5)
            
            # 일반 정보
            status_txt = f"J3:{curr_j[2]:.1f} | J5:{curr_j[4]:.1f} | Rad:{curr_radius:.0f}"
            cv2.putText(annotated_frame, status_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Singularity Safe Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event = True
                break

    except Exception as e:
        print(f"Error: {e}")
        stop_event = True
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()