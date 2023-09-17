import socket

def send_req():
    # Server address and port
    server_address = ("bias_bounty_security_container", 9999)

    # Create a socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    try:
        client_socket.connect(server_address)
    except ConnectionRefusedError:
        return "Server is not available, contact administrator"

    with client_socket:
        # Send a request message
        request_message = "Req"
        client_socket.send(request_message.encode())

        # Set a timeout of 30 seconds
        client_socket.settimeout(30)

        try:
            # Receive the response from the server
            response = client_socket.recv(1024).decode()
            return response
        except socket.timeout:
            return "Function forward pass taking too long, exiting for security measures"

if __name__ == "__main__":
    send_req()