#main

import os
from flask import Flask, jsonify
from socketio_instance import socketio
from register import register_bp
from recognition import recognition_bp
from notification_ws import NotificationNamespace, notification_ws

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key')  # 실제 비밀 키로 변경하세요.

# SocketIO 인스턴스 초기화 (Flask 앱과 연결)
socketio.init_app(app)


app.register_blueprint(register_bp, url_prefix='/api')
app.register_blueprint(recognition_bp, url_prefix='/api')
app.register_blueprint(notification_ws, url_prefix='/api')


socketio.on_namespace(NotificationNamespace('/notifications'))

# 임시 테스트 라우트
@app.route('/test_emit')
def test_emit():
    from notification_ws import test_emit_recognition_result
    test_emit_recognition_result()
    return jsonify({"status": "success", "message": "Test emit sent."})

if __name__ == "__main__":
   
    socketio.run(app, host="0.0.0.0", port=5001)
