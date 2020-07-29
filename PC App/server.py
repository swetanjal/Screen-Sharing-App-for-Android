from http.server import HTTPServer, BaseHTTPRequestHandler
import random
import glob
import os
class HelloHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # First, send a 200 OK response.
        self.send_response(200)
        # Then send headers.
        self.send_header('Content-type', 'image/jpg')
        self.end_headers()

        # Now, write the response body.
        for i in range(100):
            os.system("import -window root " + str(i) + ".jpg")
            self.wfile.write(load_binary(str(i) + ".jpg"))
        # self.wfile.write(load_binary(files[random.randint(0, len(files) - 1)] ))
        # self.wfile.write(load_binary(files[random.randint(0, len(files) - 1)] ))

def load_binary(file):
    with open(file, 'rb') as file:
        return file.read()

if __name__ == '__main__':
    server_address = ('', 8000)  # Serve on all addresses, port 8000.
    httpd = HTTPServer(server_address, HelloHandler)
    httpd.serve_forever()