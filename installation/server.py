#!/usr/bin/python3

import socket
import subprocess
import security
def listen():
    # Server address and port
    server_address = ("0.0.0.0", 9999)

    # Create a socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the server address
    server_socket.bind(server_address)

    # Listen for incoming connections (1 connection at a time)
    server_socket.listen(1)

    while True:
        # Accept incoming connection
        client_socket, client_address = server_socket.accept()

        print(f"Received connection from {client_address}")

        # Receive the request message
        client_socket.recv(8).decode()
        try:
            # Check if the request is valid
            output = security.main()
        except:
            output = "Error in security procedure"
        # Send the output back to the client
        client_socket.send(output.encode())

        # Close the client socket
        client_socket.close()

if __name__ == "__main__":
    listen()