
import zmq
import json

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("ipc:///tmp/feeds/0")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print(f"Sending request {request} ...")
    socket.send_string(json.dumps(["Hello", "World"]))

    #  Get the reply.
    message = socket.recv()
    print(f"Received reply {request} [ {message} ]")
