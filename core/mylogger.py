import datetime as dt
import json
import logging
import logging.config
import logging.handlers
import json
import pathlib
import os


class BenchmarkParentFilter(logging.Filter):
    """
    Allows all logs from the 'benchmark' logger at INFO+,
    but only allows ERROR+ from 'benchmark.xxx' child loggers.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "benchmark":
            # This is the parent itself: allow everything >= INFO
            return True
        if record.name.startswith("benchmark."):
            # This is a child logger: allow only ERROR or CRITICAL
            return record.levelno >= logging.ERROR
        # If it's not benchmark or a child, do not pass it to this handler
        return False


class ErrorCriticalPropagationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Only allow ERROR or CRITICAL to propagate
        return record.levelno >= logging.ERROR


class MyJSONFormatter(logging.Formatter):
    def __init__(self, *, fmt_keys=None):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys else {}

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        # ...
        # (same code you had before, omitted here for brevity)
        import datetime as dt
        import json
        
        LOG_RECORD_BUILTIN_ATTRS = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }

        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }
        if record.exc_info:
            always_fields["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: (
                msg_val if (msg_val := always_fields.pop(val, None)) else getattr(record, val)
            )
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


def setup_logging_benchmark():
    current_path = os.path.dirname(os.path.realpath(__file__))
    config_file = pathlib.Path(os.path.join(current_path, "../logs/logging_configs/benchmark.json"))
    with open(config_file) as f_in:
        config = json.load(f_in)

    # -----------------------------------------------------------------
    # Create directories for file-based handlers if they don't exist
    # -----------------------------------------------------------------
    for handler_name, handler_info in config.get("handlers", {}).items():
        filename = handler_info.get("filename", None)
        if filename:
            log_path = pathlib.Path(filename)
            # Create parent directory if needed
            log_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply the logging configuration
    logging.config.dictConfig(config)

    # -----------------------------------------------------------------
    # Attach our BenchmarkParentFilter to the parent’s handlers
    # so that only ERROR+ from children are allowed to pass.
    # -----------------------------------------------------------------
    benchmark_logger = logging.getLogger("benchmark")
    for handler in benchmark_logger.handlers:
        handler.addFilter(BenchmarkParentFilter())

    # If you had some reason to filter child loggers themselves, you could do so here,
    # but typically you only need the parent’s handlers to filter out child messages 
    # below ERROR level.


def setup_logging_benchmark_multiprocess():
    current_path = os.path.dirname(os.path.realpath(__file__))
    config_file = pathlib.Path(os.path.join(current_path, "../logs/logging_configs/benchmark-multiprocess.json"))
    with open(config_file) as f_in:
        config = json.load(f_in)

    # Pre-create directories if needed
    for handler_name, handler_info in config.get("handlers", {}).items():
        filename = handler_info.get("filename")
        if filename:
            log_path = pathlib.Path(filename)
            log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)

    # If you still want to filter child logs, attach your custom filter
    benchmark_logger = logging.getLogger("benchmark")
    for handler in benchmark_logger.handlers:
        handler.addFilter(BenchmarkParentFilter())