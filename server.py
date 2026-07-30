from http.server import SimpleHTTPRequestHandler, HTTPServer
import os


class CustomHandler(SimpleHTTPRequestHandler):
    def guess_type(self, path):
    
        if path.endswith('.js'):
            return 'application/javascript'
        
        return super().guess_type(path)

os.chdir('static')  

# Start the server
PORT = 5500
server = HTTPServer(('localhost', PORT), CustomHandler)
print(f"Serving on http://localhost:{PORT}")
server.serve_forever()