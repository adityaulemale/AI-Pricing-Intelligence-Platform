import sys
from src.logger import logger


def error_message_detail(error: Exception, error_detail: sys):
    """
    Return detailed information about an exception,
    including the file name and line number.
    """

    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    error_message = (
        f"Error occurred in Python script: [{file_name}] "
        f"at line number: [{line_number}] "
        f"with error message: [{str(error)}]"
    )

    return error_message


class CustomException(Exception):

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))

        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

    def __str__(self):
        return self.error_message

