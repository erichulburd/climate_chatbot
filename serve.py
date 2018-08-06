from flask import Flask, send
from flask_socketio import SocketIO
from trainer.serve import ModelServer
import os

app = Flask(__name__)
socketio = SocketIO(app)

model_server = ModelServer(os.getenv('job_id'))

@socketio.on('message')
def handle_message(message):
  responses = model_server.respond([message], top=1)
  send(responses[0][0])


if __name__ == '__main__':
  socketio.run(app)
