"""
Small shared utility helpers.
"""


def format_duration(total_seconds):
    """Format a duration in seconds as hours, minutes, and seconds."""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"
