# notification_ws.py

from flask import Blueprint
from flask_socketio import emit, Namespace, join_room, leave_room
import time
from socketio_instance import socketio
import threading

notification_ws = Blueprint('notification_ws', __name__)

SEND_INTERVAL = 10  # 전송 간격을10
last_sent_times = {}  
lock = threading.Lock()

def background_emit(recognized_id, recognized_name):
    """
    백그라운드에서 단일 인식 결과를 클라이언트로 전송.
    """
    socketio.emit(
        'recognition_result',
        {
            "status": "success",
            "result": {
                "id": recognized_id,
                "name": recognized_name
            }
        },
        namespace='/notifications'
    )
    print(f"[background_emit] Successfully sent recognition_result: ID={recognized_id}, Name={recognized_name}")

def send_recognition_result(user_id, name):
    """
    인식 결과를 모든 연결된 클라이언트에 전송 (사용자별 레이트 제한).
    """
    global last_sent_times
    current_time = time.time()
    with lock:
        last_sent = last_sent_times.get(user_id, 0)
        if current_time - last_sent >= SEND_INTERVAL:
            last_sent_times[user_id] = current_time
            print(f"[send_recognition_result] Preparing to send recognition_result: ID={user_id}, Name={name}")
            socketio.start_background_task(background_emit, user_id, name)
        else:
            print(f"[send_recognition_result] Not sending recognition_result for ID={user_id}, Name={name} due to rate limiting.")

def update_recognition_status(success, recognized_id="", recognized_name=""):

    if success:
        print(f"[update_recognition_status] Recognition successful for: ID={recognized_id}, Name={recognized_name}")
        send_recognition_result(recognized_id, recognized_name)
    else:
        print("[update_recognition_status] Recognition failed.")

class NotificationNamespace(Namespace):
    def on_connect(self):
        
        print("클라이언트가 /notifications 네임스페이스에 연결되었습니다.")
        emit('response', {'message': 'Notification 네임스페이스에 연결되었습니다.'})
        join_room('all')  # 모든 클라이언트를 'all' 룸에 참여시킴

    def on_disconnect(self):
        
        print("클라이언트가 /notifications 네임스페이스에서 연결을 해제했습니다.")
        leave_room('all')

    def on_custom_event(self, data):
        
        print(f"클라이언트로부터 받은 데이터: {data}")
        emit('response', {'message': '커스텀 이벤트가 성공적으로 처리되었습니다.'})

def test_emit_recognition_result():
    """테스트를 위해 임시로 인식 결과를 전송."""
    test_id = 1234
    test_name = "eunjin"
    print(f"[test_emit_recognition_result] Sending recognition_result: ID={test_id}, Name={test_name}")
    socketio.start_background_task(background_emit, test_id, test_name)
