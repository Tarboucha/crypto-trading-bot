import os
import shutil

from rich.console import Console


def _get_terminal_width() -> int:
    """Detect terminal width, default to 200 in non-interactive environments."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return 200
    return shutil.get_terminal_size().columns


# Main console for standard output
console = Console(width=_get_terminal_width())

# Separate console for logging (stderr), so logs don't mix with program output
error_console = Console(stderr=True, width=_get_terminal_width())
