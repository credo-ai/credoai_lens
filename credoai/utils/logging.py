from collections import deque
from io import StringIO
from logging import FileHandler, Formatter, Handler, StreamHandler, getLogger
from os.path import join
from sys import stdout


class TailLogHandler(Handler):
    def __init__(self, log_queue):
        Handler.__init__(self)
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.append(self.format(record))


class TailLogger(object):
    def __init__(self, maxlen):
        self._log_queue = deque(maxlen=maxlen)
        self._log_handler = TailLogHandler(self._log_queue)

    def contents(self):
        return "\n".join(self._log_queue)

    @property
    def log_handler(self):
        return self._log_handler


class Logger:
    def __init__(
        self, name, path=None, record_stream=True, logging_level="info", formatter=None
    ):
        self.file_path = None
        self.stream = None
        if formatter is None:
            formatter = Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        self.formatter = formatter
        self.logger = self.setup_logger(name, logging_level)
        self.log_capture_string = StringIO()
        if record_stream:
            self.stream = self.setup_stream()
        if path:
            self.file_path = join(path, f"{name}.log")
            self.setup_file()

    def setup_logger(self, name, logging_level):
        logger = getLogger(name)
        handler = StreamHandler(stdout)
        handler.setFormatter(self.formatter)
        logger.addHandler(handler)
        logger.setLevel(logging_level)
        return logger

    def setup_stream(self, tail_length=100):
        tail = TailLogger(tail_length)
        log_handler = tail.log_handler
        log_handler.setFormatter(self.formatter)
        self.logger.addHandler(log_handler)
        return tail

    def setup_file(self):
        file_handler = FileHandler(self.file_path)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)


def setup_logger(name="lens", path=None, record_stream=False, logging_level="INFO"):
    tmp = Logger(name, path, record_stream, logging_level)
    return tmp.logger, tmp.stream


global_logger, global_tail = setup_logger(path=".")
