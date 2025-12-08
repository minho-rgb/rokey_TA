import cv2

cap = cv2.VideoCapture(6)           # 웹캡 또는 비디오 소스 열기
tracker = cv2.TrackerCSRT_create() # CSRT 트래커 생성
init_once = False

while True:
    ret, frame = cap.read()
    if not ret:                     # 프레임 읽기 실패 시 종료
        break

    if not init_once:
        # 첫 프레임에서 추적할 객체의 ROI 선택 (마우스로 영역 지정)
        bbox = cv2.selectROI("Frame", frame, False, False)
        tracker.init(frame, bbox)   # 선택한 영역으로 트래커 초기화
        init_once = True
    else:
        # 이후 프레임에서 객체 추적
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            # 추적된 객체 주위에 초록색 바운딩 박스 그리기
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == 27:       # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
