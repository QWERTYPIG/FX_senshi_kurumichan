import argparse
import json
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .core import Manager, ROOT


def handler_for(manager):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, kind='application/json; charset=utf-8'):
            data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if path in assets:
                name, kind = assets[path]
                return self.reply(200, (ROOT / 'manager' / 'web' / name).read_bytes(), kind)
            if path == '/assest/kurumi.jpg':
                return self.reply(200, (ROOT / 'assest' / 'kurumi.jpg').read_bytes(), 'image/jpeg')
            if path == '/api/status':
                return self.reply(200, manager.snapshot())
            pieces = path.strip('/').split('/')
            if len(pieces) == 4 and pieces[:2] == ['api', 'runs'] and pieces[3] in {'log', 'csv'}:
                with manager.lock:
                    record = next((r for r in manager.state['runs'] if r['id'] == pieces[2]), None)
                if record:
                    filename = 'run.log' if pieces[3] == 'log' else 'prediction.csv'
                    target = manager.store / 'runs' / record['id'] / filename
                    if target.is_file():
                        return self.reply(200, target.read_bytes(), 'text/plain; charset=utf-8')
            self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            # Require same-origin JSON. Cross-site forms cannot issue model commands.
            origin = self.headers.get('Origin')
            expected = f'http://{self.headers.get("Host")}'
            if origin and origin != expected:
                return self.reply(403, {'error': 'Origin rejected'})
            if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                return self.reply(403, {'error': 'Cross-site request rejected'})
            if self.headers.get_content_type() != 'application/json':
                return self.reply(415, {'error': 'JSON required'})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if size < 0 or size > 8192:
                    raise ValueError('Request too large')
                body = json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(body, dict):
                    raise ValueError('JSON object required')
                if self.path == '/api/submit-preview':
                    try:
                        result = manager.submit_selected(dry_run=True)
                    except ValueError:
                        raise
                    except Exception as exc:
                        return self.reply(502, {'error': f'檢查提交失敗：{exc}'})
                    return self.reply(200, result)
                if self.path == '/api/config':
                    manager.configure(body.get('selected'), body.get('autoSubmit'))
                    return self.reply(200, {'ok': True})
                action = self.path.removeprefix('/api/')
                if self.path.startswith('/api/') and action in {'refresh', 'state', 'train', 'predict', 'submit'}:
                    model = body.get('model')
                    if action in {'train', 'predict'} and not isinstance(model, str):
                        raise ValueError('Model required')
                    queued = manager.enqueue(action, model if action in {'train', 'predict'} else None)
                    return self.reply(202, {'queued': queued})
                self.reply(404, {'error': 'Not found'})
            except (ValueError, TypeError, queue.Full) as exc:
                self.reply(400, {'error': str(exc) or 'Queue full'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description='Local ML competition manager')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--command-timeout', type=int, default=7200)
    args = parser.parse_args()
    manager = Manager(command_timeout=args.command_timeout)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(manager))
    manager.start()
    print(f'Manager: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
