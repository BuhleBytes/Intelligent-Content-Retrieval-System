"""
Terminal UI Module
Beautiful terminal output utilities

This module provides functions for creating visually appealing terminal output
including colors, formatting, progress bars, and boxes.

Author: Buhle Mlandu
Date: December 2024
"""
import shutil

# ============================================================================
# COLOR CONSTANTS
# ============================================================================

BRIGHT_RED = "\033[91m"
BRIGHT_YELLOW = "\033[93m"
GREY = "\033[90m"
BRIGHT_CYAN = "\033[96m"
WHITE = "\033[37m"   
BRIGHT_GREEN = "\033[92m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

# ============================================================================
# BASIC UTILITIES
# ============================================================================

def terminal_width():
    """Get the width of the terminal window."""
    return shutil.get_terminal_size().columns


def centre(text, color=""):
    """
    Print text centered in the terminal.
    
    Args:
        text: Text to center
        color: Optional color code
    """
    width = terminal_width()
    start = (width - len(text)) // 2
    print(" " * start + color + text + RESET)


# ============================================================================
# HEADERS AND SECTIONS
# ============================================================================
def print_header(text):
    """
    Print a main header with borders.
    
    Args:
        text: Header text
    """
    width = terminal_width()
    print()
    print(BRIGHT_CYAN + "=" * width + RESET)
    centre(BOLD + text, BRIGHT_GREEN)
    print(BRIGHT_CYAN + "=" * width + RESET)
    print()


def print_subheader(text):
    """
    Print a subheader with dashed borders.
    
    Args:
        text: Subheader text
    """
    width = terminal_width()
    print(BRIGHT_YELLOW + "-" * width + RESET)
    centre(text, BRIGHT_YELLOW)
    print(BRIGHT_YELLOW + "-" * width + RESET)


# ============================================================================
# STATUS MESSAGES
# ============================================================================

def print_step(emoji, label, status, message=""):
    """
    Print a step with emoji, label, and status.
    
    Args:
        emoji: Emoji icon for the step
        label: Step description
        status: One of 'success', 'error', 'warning', 'info'
        message: Additional message
    
    Example:
        print_step("🤖", "Checking robots.txt", "success", "Scraping allowed")
    """
    width = terminal_width()
    
    # Left side: emoji + label
    left_part = f"{emoji}  {BRIGHT_CYAN}{label}{RESET}"
    
    # Right side: status + message
    if status == "success":
        right_part = f"{BRIGHT_GREEN}✓ {message}{RESET}"
    elif status == "error":
        right_part = f"{BRIGHT_RED}✗ {message}{RESET}"
    elif status == "warning":
        right_part = f"{BRIGHT_YELLOW}⚠ {message}{RESET}"
    elif status == "info":
        right_part = f"{GREY}ℹ {message}{RESET}"
    else:
        right_part = f"{WHITE}{message}{RESET}"
    
    # Calculate spacing
    left_len = len(label) + 4  # emoji (2) + spaces
    right_len = len(message) + 3  # icon + space
    dots_count = width - left_len - right_len - 10
    dots = GREY + "." * max(dots_count, 3) + RESET
    
    print(f"{left_part} {dots} {right_part}")


def print_success(message):
    """Print a success message."""
    print(f"{BRIGHT_GREEN}✓ {message}{RESET}")


def print_error(message):
    """Print an error message."""
    print(f"{BRIGHT_RED}✗ {message}{RESET}")


def print_warning(message):
    """Print a warning message."""
    print(f"{BRIGHT_YELLOW}⚠ {message}{RESET}")


def print_info(message):
    """Print an info message."""
    print(f"{GREY}ℹ {message}{RESET}")


# ============================================================================
# PROGRESS INDICATORS
# ============================================================================

def print_progress_bar(current, total, label="Progress"):
    """
    Print a progress bar.
    
    Args:
        current: Current progress value
        total: Total value
        label: Label for the progress bar
    
    Example:
        print_progress_bar(3, 4, "Overall Progress")
    """
    width = terminal_width()
    bar_width = min(50, width - 30)
    
    progress = current / total
    filled = int(bar_width * progress)
    
    bar = BRIGHT_GREEN + "█" * filled + GREY + "░" * (bar_width - filled) + RESET
    percentage = f"{BRIGHT_CYAN}{int(progress * 100)}%{RESET}"
    
    print(f"\r{BRIGHT_YELLOW}{label}:{RESET} {bar} {percentage} ({current}/{total})", end="")
    
    if current == total:
        print()  # New line when complete


# ============================================================================
# BOXES AND TABLES
# ============================================================================

def print_box(title, content_dict):
    """
    Print a box with title and key-value pairs.
    
    Args:
        title: Box title
        content_dict: Dictionary of key-value pairs to display
    
    Example:
        print_box("Statistics", {"Total": "1000", "Average": "250"})
    """
    width = min(70, terminal_width() - 4)
    
    print()
    print(BRIGHT_CYAN + "┌" + "─" * (width - 2) + "┐" + RESET)
    
    # Title
    title_text = f" {title} "
    padding = (width - len(title_text) - 2) // 2
    print(BRIGHT_CYAN + "│" + RESET + " " * padding + BRIGHT_GREEN + BOLD + title_text + RESET + " " * padding + BRIGHT_CYAN + "│" + RESET)
    
    print(BRIGHT_CYAN + "├" + "─" * (width - 2) + "┤" + RESET)
    
    # Content
    for key, value in content_dict.items():
        key_str = f"{BRIGHT_YELLOW}{key}:{RESET}"
        val_str = f"{WHITE}{value}{RESET}"
        line = f"  {key_str} {val_str}"
        print(BRIGHT_CYAN + "│" + RESET + line + " " * (width - len(key) - len(str(value)) - 5) + BRIGHT_CYAN + "│" + RESET)
    
    print(BRIGHT_CYAN + "└" + "─" * (width - 2) + "┘" + RESET)
    print()


# ============================================================================
# SIMPLE FORMATTING (from your original file)
# ============================================================================

def left():
    """Print a prompt on the left side."""
    print(BRIGHT_RED + "You" + GREY + " > ", end="")


def right(text, l, m, r):
    """
    Print text on the right side of terminal.
    
    Args:
        text: Full text (for length calculation)
        l: Left part (cyan)
        m: Middle part (grey)
        r: Right part (white)
    """
    width = terminal_width()
    start_position = width - len(text)
    print("\r" + " " * start_position, end="")
    print(BRIGHT_CYAN + str(l) + RESET, end="")
    print(GREY + str(m) + RESET, end="")
    print(WHITE + str(r), sep="")
    print(BRIGHT_RED + "\nYou " + GREY + "> ", end="")


def default():
    """Reset terminal colors to default."""
    print(RESET)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_line():
    """Clear the current line."""
    print("\r" + " " * terminal_width() + "\r", end="")


def print_separator(char="─"):
    """Print a full-width separator line."""
    print(GREY + char * terminal_width() + RESET)