from wsgiref.simple_server import make_server, WSGIServer
from socketserver import ThreadingMixIn


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class Server:
    def __init__(self, wsgi_app, host='0.0.0.0', port=8070):
        self.wsgi_app = wsgi_app
        self.host = host
        self.port = port
        self.server = make_server(self.host, self.port, self.wsgi_app,
                                  ThreadingWSGIServer)

    def serve_forever(self):
        print("Listening on http://{}:{}/".format(self.host, self.port))
        print("Hit Ctrl-C to quit.")
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.server.server_close()
