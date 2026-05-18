import sys


def _error_message(error: Exception, error_detail: sys) -> str:
    """Build a message that includes file name, line number, and the original error."""
    _, _, tb = error_detail.exc_info()
    filename = tb.tb_frame.f_code.co_filename
    lineno = tb.tb_lineno
    return f"Error in [{filename}] at line [{lineno}]: {str(error)}"


class CustomException(Exception):
    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(str(error))
        self.error_message = _error_message(error, error_detail)

    def __str__(self) -> str:
        return self.error_message
