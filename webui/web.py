import http.server
import socketserver
from pathlib import Path

PORT = 8080

# 设置服务器根目录
root_dir = Path(__file__).parent

# 自定义请求处理类
class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(root_dir), **kwargs)

    def do_GET(self):
        # 如果访问根路径，则返回 btc_index.html
        if self.path == '/':
            self.path = '/btc_index.html'
        return super().do_GET()

# 启动服务器
with socketserver.TCPServer(("", PORT), MyHttpRequestHandler) as httpd:
    print(f"服务器启动在端口 {PORT}")
    httpd.serve_forever()
