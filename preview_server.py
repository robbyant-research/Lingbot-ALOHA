"""Local static preview with byte ranges for MP4 seeking. Python 3, no dependencies."""
import argparse
import functools
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        path = self.translate_path(self.path)
        header = self.headers.get('Range')
        if not header or not os.path.isfile(path):
            return super().send_head()
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
        size = os.path.getsize(path)
        if not match or not any(match.groups()) or size == 0:
            return self.invalid_range(size)
        first, last = match.groups()
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        else:
            start = max(0, size - int(last))
            end = size - 1
        if start >= size or start > end or (not first and int(last) == 0):
            return self.invalid_range(size)
        stream = open(path, 'rb')
        stream.seek(start)
        self.remaining = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header('Content-type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(self.remaining))
        self.send_header('Last-Modified', self.date_time_string(os.fstat(stream.fileno()).st_mtime))
        self.end_headers()
        return stream

    def invalid_range(self, size):
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header('Content-Range', f'bytes */{size}')
        self.send_header('Content-Length', '0')
        self.end_headers()
        return None

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            while self.remaining:
                data = source.read(min(65536, self.remaining))
                if not data:
                    break
                outputfile.write(data)
                self.remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--directory', default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=args.directory)
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'HALO preview: http://127.0.0.1:{args.port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
