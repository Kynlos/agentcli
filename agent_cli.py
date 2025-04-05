#!/usr/bin/env python3
"""
AgentCLI: An autonomous AI assistant for file operations, command execution,
web browsing, code execution, and more, operating in a think-act-observe loop.
"""
import base64
import os
import sys
import json
import time
import subprocess
import platform
import re
import traceback
import glob
import hashlib
import io
import tempfile
import threading
import queue
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import shlex # Import shlex (though not used in final parser, good to keep in mind)
import codecs # Import codecs for escape sequence handling

# --- Dependency Handling ---
# Handle readline platform differences
if platform.system() != 'Windows':
    try:
        import readline
    except ImportError:
        print("readline module not available on this system.")

# Check for colorama and install if needed
try:
    from colorama import init, Fore, Style, Back
except ImportError:
    print("Installing colorama...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
    from colorama import init, Fore, Style, Back

# Check for tqdm (progress bars) and install if needed
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

# Check for Pillow (image processing) and install if needed
try:
    from PIL import Image
except ImportError:
    print("Installing Pillow for image processing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

# Check for google-generativeai and install if needed
try:
    import google.generativeai as genai
    # *** Import types explicitly for specific configs/enums ***
    from google.generativeai import types

    # *** Verify required attributes under 'types' (excluding Content/Part) ***
    required_attrs_in_types = ["GenerationConfig", "HarmCategory", "HarmBlockThreshold", "SafetySetting"] # Added SafetySetting based on usage
    missing_attrs_in_types = [attr for attr in required_attrs_in_types if not hasattr(types, attr)]
    if missing_attrs_in_types:
        print(f"{Fore.YELLOW}Warning: Could not find expected attributes in google.generativeai.types: {', '.join(missing_attrs_in_types)}. API might have changed.{Style.RESET_ALL}")

    # *** Verify GenerativeModel is directly under 'genai' - that's all we need ***
    required_attrs_in_genai = ["GenerativeModel"] # Only GenerativeModel is essential
    missing_attrs_in_genai = [attr for attr in required_attrs_in_genai if not hasattr(genai, attr)]
    if missing_attrs_in_genai:
         # This would be a critical error if GenerativeModel is missing
        print(f"{Fore.RED}Error: Could not find critical attributes in google.generativeai: {', '.join(missing_attrs_in_genai)}. Library might be corrupted or incompatible.{Style.RESET_ALL}")
        # sys.exit(1) # Optionally exit if fundamental classes are missing


except ImportError as e:
    print(f"Installing/updating google-generativeai... ({e})")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "google-generativeai"])
    import google.generativeai as genai
    # *** Import types explicitly after install ***
    from google.generativeai import types


# Check for requests and install if needed
try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

# Initialize colorama
init(autoreset=True)

# --- Configuration ---
CONFIG = {
    # Server connections
    "memory_server_url": "http://localhost:8000",  # Server-memory MCP server
    "puppeteer_server_url": "http://localhost:3000",  # Puppeteer MCP server
    "memory_id": "agent_cli_memory_v3",  # Unique ID for this agent's memory
    "memory_server_enabled": True,  # Set to False if server not available
    "puppeteer_server_enabled": True,  # Set to False if server not available
    "memory_server_timeout": 5,  # Timeout in seconds for memory server connections
    "auto_retry_memory_connection": False, # Whether to retry memory server connection on failure

    # Model configuration
    "model": "gemini-2.5-pro-exp-03-25",  # Default model
    "model_temperature": 0.4,  # Temperature for generation (0.0-1.0)
    "model_max_output_tokens": 8192,  # Maximum output tokens

    # Execution settings
    "max_iterations_per_request": 50,  # Limit autonomous loop iterations
    "confirm_risky_actions": True,  # Ask before executing EXECUTE_COMMAND or DELETE_FILE
    "command_timeout": 120,  # Timeout for command execution in seconds
    "max_file_size_read": 5 * 1024 * 1024,  # 5MB max file size for reading
    "max_file_size_download": 50 * 1024 * 1024,  # 50MB max file size for downloads

    # Performance settings
    "enable_response_cache": True,  # Cache API responses to reduce token usage
    "cache_expiry_seconds": 3600,  # Cache expiry time in seconds (1 hour)
    "enable_parallel_processing": True,  # Enable parallel processing for certain operations
    "max_worker_threads": 4,  # Maximum number of worker threads for parallel operations

    # UI settings
    "show_progress_bars": True,  # Show progress bars for long operations
    "enable_command_history": True,  # Enable command history and recall
    "max_command_history": 100,  # Maximum number of commands to keep in history
    "enable_logging": True,  # Enable logging to file
    "log_file": "agent_cli.log",  # Log file name
    "log_level": "INFO",  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    # New feature flags
    "enable_code_execution": True,  # Enable Python/JS code execution
    "enable_image_generation": False,  # Enable image generation (requires additional API)
    "enable_file_diff": True,  # Enable file diff visualization
    "enable_clipboard": True,  # Enable clipboard operations
}

# --- Global Memory ---
memory = {
    "conversation": [],
    "observations": [],
    "files": {},
    "browser_state": None,
    "command_history": [],
    "code_execution": {},
    "downloads": {},
    "search_results": {},
    "clipboard": None
}

# --- Response Cache ---
response_cache = {}

# --- Logging Setup ---
def setup_logging():
    """Set up logging to file if enabled"""
    if not CONFIG.get("enable_logging", False):
        return

    import logging
    log_file = CONFIG.get("log_file", "agent_cli.log")
    log_level_str = CONFIG.get("log_level", "INFO")
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"AgentCLI logging started. Version: 2.0.0")

# --- Helper Functions ---
def format_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def get_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def run_with_progress(func, *args, desc="Processing", **kwargs):
    """Run a function with a progress bar"""
    if not CONFIG.get("show_progress_bars", True):
        return func(*args, **kwargs)

    result_queue = queue.Queue()

    def worker():
        try:
            result = func(*args, **kwargs)
            result_queue.put((True, result))
        except Exception as e:
            result_queue.put((False, e))

    thread = threading.Thread(target=worker)
    thread.start()

    with tqdm(desc=desc, unit="", ncols=80) as pbar:
        while thread.is_alive():
            pbar.update(1)
            thread.join(0.1)

    success, result = result_queue.get()
    if success:
        return result
    else:
        raise result  # Re-raise the exception

# --- Action Result Structure ---
class ActionResult:
    def __init__(self, status: str, output: Optional[str] = None, error: Optional[str] = None, data: Any = None):
        self.status = status # 'success' or 'error'
        self.output = output # Standard output or success message
        self.error = error   # Error message or stderr
        self.data = data     # Any additional structured data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "data": self.data
        }

    def __str__(self) -> str:
        parts = [f"Status: {self.status}"]
        if self.output: parts.append(f"Output: {self.output[:200]}{'...' if len(self.output) > 200 else ''}")
        if self.error: parts.append(f"Error: {self.error}")
        if self.data: parts.append(f"Data: {str(self.data)[:200]}{'...' if len(str(self.data)) > 200 else ''}")
        return "\n".join(parts)

# --- Helper Function for Parsing Parameters ---
def _parse_key_value_parameters(params_str: str) -> Dict[str, str]:
    """
    Parses a string of key=value parameters, handling quotes and standard escapes.
    Improved to handle multi-line quoted strings better and interpret standard
    escape sequences (like \n, \t) within quoted values.
    Returns a dictionary.
    """
    params = {}
    # Using a simpler regex that might be more robust for typical agent inputs
    # It looks for key=value pairs, handling simple quotes.
    # Adjusted to handle potential lack of comma separation if it's the last parameter.
    pattern = re.compile(r"""
        \s*                         # Optional leading whitespace
        ([a-zA-Z_][a-zA-Z0-9_]*)     # Key name (starts with letter or _, then alphanumeric or _)
        \s*=\s*                     # Equals sign, surrounded by optional whitespace
        (                           # Start value capture group
            "(?P<double_quoted_val> (?:\\.|[^"\\])* )"  # Double-quoted value, handles escaped quotes \"
            |
            '(?P<single_quoted_val> (?:\\.|[^'\\])* )'  # Single-quoted value, handles escaped quotes \'
            |
            (?P<unquoted_val>        # Unquoted value
                (?:\\.|[^\s,"'=])    # Either an escaped character or any character that isn't whitespace, comma, quote, or equals
                [^,]*?               # Non-greedily match anything until a comma (or end)
            )
        )                           # End value capture group
        \s*                         # Optional trailing whitespace
        (?:,|$)                     # Match a comma or the end of the string
    """, re.VERBOSE | re.DOTALL)

    pos = 0
    while pos < len(params_str):
        match = pattern.match(params_str, pos)
        if match:
            key = match.group(1)
            value = None
            raw_value = "" # Store the raw matched value before unescaping

            if match.group("double_quoted_val") is not None:
                raw_value = match.group("double_quoted_val")
                # 1. Unescape quotes and literal backslashes specific to the regex capture
                value = raw_value.replace('\\"', '"').replace('\\\\', '\\')
                # 2. Unescape standard Python string escapes (like \n, \t)
                try:
                    value = codecs.decode(value, 'unicode_escape')
                except UnicodeDecodeError as ude:
                    print(f"{Fore.YELLOW}Warning: Failed to unicode-unescape double-quoted value for key '{key}'. Error: {ude}. Using raw value after basic unescaping.{Style.RESET_ALL}")
                    # Fallback to value with only quotes/backslashes unescaped
            elif match.group("single_quoted_val") is not None:
                raw_value = match.group("single_quoted_val")
                # 1. Unescape quotes and literal backslashes specific to the regex capture
                value = raw_value.replace("\\'", "'").replace('\\\\', '\\')
                # 2. Unescape standard Python string escapes (like \n, \t)
                try:
                    value = codecs.decode(value, 'unicode_escape')
                except UnicodeDecodeError as ude:
                    print(f"{Fore.YELLOW}Warning: Failed to unicode-unescape single-quoted value for key '{key}'. Error: {ude}. Using raw value after basic unescaping.{Style.RESET_ALL}")
                    # Fallback to value with only quotes/backslashes unescaped
            elif match.group("unquoted_val") is not None:
                raw_value = match.group("unquoted_val")
                # For unquoted values, only strip whitespace. Do NOT apply unicode_escape
                # as it might misinterpret literal backslashes (e.g., in paths).
                value = raw_value.strip()
                # If needed, specific manual unescaping for unquoted could go here, but it's fragile.

            params[key] = value if value is not None else ""
            pos = match.end()
            # Check if the matched part ended with the end of the string ($) or a comma
            # If it ended with a comma, advance past it
            if pos < len(params_str) and params_str[pos-1] == ',':
                 # If the character *before* current position was a comma,
                 # we are already positioned after it by match.end().
                 pass
            elif not params_str[pos:].strip():
                 break # End of string reached
            # Check if the next non-whitespace char is a comma, if so, consume it
            next_char_match = re.match(r'\s*,', params_str[pos:])
            if next_char_match:
                pos += next_char_match.end()

        else:
            # If no match, check if there's just trailing whitespace or comma
            remainder = params_str[pos:].strip()
            if not remainder or remainder == ',':
                break
            # Report error if significant text remains unmatched
            print(f"{Fore.RED}Error: Parameter parsing failed near position {pos}: '{params_str[pos:pos+50]}...'{Style.RESET_ALL}")
            # Attempt recovery: Skip until the next potential start of a parameter (e.g., comma)
            next_comma = params_str.find(',', pos)
            if next_comma != -1:
                 pos = next_comma + 1
                 print(f"{Fore.YELLOW}Attempting recovery, skipping to position {pos}{Style.RESET_ALL}")
            else:
                 break # No more commas, assume end of parameters

    return params


# --- Agent Class ---
class AgentCLI:
    def __init__(self):
        self.setup_gemini()
        self.load_memory()
        self.running = True
        self.system_prompt = self._build_system_prompt() # Build system prompt once

    def _build_system_prompt(self):
        """Builds the system prompt string."""
        # Update known files and browser state dynamically within the prompt string
        known_files_list = list(memory['files'].keys())
        browser_state_str = str(memory.get('browser_state', 'None')) # Convert state to string
        clipboard_content = memory.get('clipboard', 'None')
        clipboard_preview = str(clipboard_content)[:50] + '...' if clipboard_content and len(str(clipboard_content)) > 50 else str(clipboard_content)

        # Build the actions list based on enabled features
        actions_list = [
            "1.  CREATE_FILE(filename: str, content: str) - Creates a new file. Content can be multi-line. **Important:** If the filename ends with `.html` or `.htm`, generate valid HTML markup in the `content`. Use `<p>` tags for paragraphs and `<br>` tags for explicit line breaks instead of relying on plaintext newlines (`\\n`). Example: CREATE_FILE(filename=\"index.html\", content=\"<h1>Title</h1><p>Hello<br>World!</p>\")",
            "2.  READ_FILE(filename: str) - Reads the entire content of a file. Example: READ_FILE(filename=\"test.txt\")",
            "3.  EDIT_FILE(filename: str, new_content: str) - Overwrites a file with new content. Creates if not exists. **Important:** If the filename ends with `.html` or `.htm`, ensure `new_content` contains valid HTML markup, using `<p>`, `<br>`, etc., appropriately. Example: EDIT_FILE(filename=\"page.html\", new_content=\"<p>Updated content.</p>\")",
            "4.  DELETE_FILE(filename: str) - Deletes a file. **Use with caution.** Example: DELETE_FILE(filename=\"test.txt\")",
            "5.  EXECUTE_COMMAND(command: str) - Executes a shell command. **Use with caution.** Output and errors will be captured. Example: EXECUTE_COMMAND(command=\"ls -l\")",
            "6.  BROWSE_WEB(url: str) - Navigates a headless browser to the URL. Updates browser state. Example: BROWSE_WEB(url=\"https://example.com\")",
            "7.  SCREENSHOT_WEB(selector: Optional[str] = None) - Takes a screenshot of the current page. Saves as PNG. If selector is provided, captures only that element. Example: SCREENSHOT_WEB(selector=\"#main\") or SCREENSHOT_WEB()",
            "8.  SEARCH_WEB(query: str) - Performs a web search (uses BROWSE_WEB to Google and SCREENSHOT_WEB). Example: SEARCH_WEB(query=\"python tutorial\")",
            "9.  LIST_FILES(directory: str = '.', pattern: str = '*') - Lists files in the specified directory matching the pattern. Example: LIST_FILES(directory=\"src\", pattern=\"*.py\")",
            "10. FIND_IN_FILES(pattern: str, directory: str = '.', file_pattern: str = '*') - Searches for a pattern in files. Example: FIND_IN_FILES(pattern=\"function main\", directory=\"src\", file_pattern=\"*.py\")",
        ]

        # Add conditional actions based on configuration
        if CONFIG.get("enable_code_execution", False):
            actions_list.append("11. EXECUTE_PYTHON(code: str) - Executes Python code and returns the result. Example: EXECUTE_PYTHON(code=\"print('Hello, world!')\")")
            actions_list.append("12. EXECUTE_JAVASCRIPT(code: str) - Executes JavaScript code in the browser context. Example: EXECUTE_JAVASCRIPT(code=\"document.title\")")

        if CONFIG.get("enable_file_diff", False):
            actions_list.append("13. DIFF_FILES(file1: str, file2: str) - Shows differences between two files. Example: DIFF_FILES(file1=\"old.txt\", file2=\"new.txt\")")

        if CONFIG.get("enable_clipboard", False):
            actions_list.append("14. COPY_TO_CLIPBOARD(text: str) - Copies text to clipboard. Example: COPY_TO_CLIPBOARD(text=\"Hello, world!\")")
            actions_list.append("15. GET_CLIPBOARD() - Gets the current clipboard content. Example: GET_CLIPBOARD()")

        actions_list.append("16. DOWNLOAD_FILE(url: str, filename: str) - Downloads a file from a URL. Example: DOWNLOAD_FILE(url=\"https://example.com/file.pdf\", filename=\"downloaded.pdf\")")
        actions_list.append("17. SUMMARIZE_TEXT(text: str) - Summarizes long text. Example: SUMMARIZE_TEXT(text=\"Long text to summarize...\")")

        # Join the actions with newlines
        actions_str = "\n".join(actions_list)

        return f"""You are AgentCLI 2.0, an advanced autonomous AI assistant. Your goal is to fulfill the user's request by thinking step-by-step and executing actions.

Current working directory: {os.getcwd()}
Files known (from previous operations): {known_files_list}
Current Browser State: {browser_state_str}
Clipboard Content: {clipboard_preview}

Available Actions (Format: ACTION_NAME(key="value", key2="value2", ...)):
{actions_str}

Your Response Format:
You MUST respond using the following structure:
<thinking>
Your step-by-step reasoning about the current state, the goal, and what action needs to be taken next. Analyze previous observations, especially errors, to correct your plan. Consider the target file type when generating content.
</thinking>
<action>
ONE SINGLE action from the list above, in the exact format ACTION_NAME(parameter_name="parameter_value", ...). Ensure parameters are correctly quoted, especially if they contain special characters or newlines. Use standard JSON-style string escaping for quotes (\\") and backslashes (\\\\) within the parameter values if needed. **Your parameters will be parsed, automatically handling escapes like \\n within quoted strings.**
Example: CREATE_FILE(filename="script.py", content="print('hello\\nworld')")
Example: EDIT_FILE(filename="data.json", new_content="{{ \\"key\\": \\"value\\" }}")
</action>

OR, if the goal is complete or you cannot proceed:
<thinking>
Explain why the task is complete or why you are stuck. Provide details if stuck.
</thinking>
<final_response>
Your final message to the user.
</final_response>

IMPORTANT RULES:
- Only output ONE action tag per response.
- Ensure actions use the `key="value"` format for parameters as shown in the examples. Pay close attention to the required key names (e.g., `new_content` for EDIT_FILE).
- **When creating or editing HTML files (`.html`, `.htm`), generate proper HTML structure (e.g., using `<p>`, `<br>`) instead of plaintext newlines (`\\n`) for formatting.**
- Use the observations from previous steps provided in the history to inform your next action. Pay close attention to error messages in observations.
- If an action fails, analyze the error in your thinking and decide whether to retry (with corrections), try a different approach, or give up.
- Do not create files or execute commands unless explicitly necessary for the user's goal.
- Before deleting files or executing commands, briefly state the reason in your thinking step.
"""

    def setup_gemini(self):
        """Set up Gemini API configuration"""
        # *** Use GEMINI_API_KEY consistently ***
        api_key = os.environ.get("GEMINI_API_KEY") # Changed from GOOGLE_API_KEY to match example
        if not api_key:
            # *** Match error message to environment variable name ***
            print(f"{Fore.RED}Fatal Error: GEMINI_API_KEY environment variable not set.{Style.RESET_ALL}")
            sys.exit(1)

        try:
            # Configure the API key for the library
            genai.configure(api_key=api_key)

            # Check if GenerativeModel exists (required for operation)
            if not hasattr(genai, 'GenerativeModel'):
                print(f"{Fore.RED}Fatal Error: google.generativeai module does not have 'GenerativeModel' attribute. Check library installation/version.{Style.RESET_ALL}")
                sys.exit(1)

            # *** Store model name for later use ***
            self.model_name = CONFIG["model"]

            # --- Model availability check (Optional but good practice) ---
            try:
                print(f"{Fore.CYAN}Checking available models...{Style.RESET_ALL}")
                # Use genai.list_models() directly if available
                if hasattr(genai, 'list_models'):
                    found = False
                    try:
                        for model_info in genai.list_models():
                            # Check for both exact match and with 'models/' prefix
                            if (model_info.name == self.model_name or
                                model_info.name.endswith(f"/{self.model_name}") or
                                f"models/{self.model_name}" == model_info.name):
                                found = True
                                # Always use the full model name for consistency
                                if not self.model_name.startswith('models/') and model_info.name.startswith('models/'):
                                    self.model_name = model_info.name
                                break

                        if found:
                            print(f"{Fore.GREEN} Configured model '{self.model_name}' is available.{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.YELLOW}Warning: Configured model '{CONFIG['model']}' not found in the model list. Will attempt to use it anyway.{Style.RESET_ALL}")
                    except Exception as list_error:
                        print(f"{Fore.YELLOW}Warning: Error while listing models: {list_error}. Will use configured model directly.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Warning: Cannot list models using genai.list_models(). Will attempt to use configured model '{self.model_name}' directly.{Style.RESET_ALL}")


            except Exception as model_error:
                print(f"{Fore.YELLOW}Warning: Could not check available models: {model_error}. Will use configured model '{self.model_name}'.{Style.RESET_ALL}")
            # --- End Model Check ---

            print(f"{Fore.CYAN} Gemini API configured for model: {self.model_name}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Fatal Error: Failed to initialize Gemini API: {e}{Style.RESET_ALL}")
            print(f"{Fore.RED}Traceback: {traceback.format_exc()}{Style.RESET_ALL}")
            sys.exit(1)


    def load_memory(self):
        """Load agent memory from server if available"""
        if not CONFIG.get("memory_server_enabled"):
            print(f"{Fore.YELLOW}! Memory server disabled, using in-memory storage only.{Style.RESET_ALL}")
            return

        url = f"{CONFIG['memory_server_url']}/memory/{CONFIG['memory_id']}"
        timeout = CONFIG.get("memory_server_timeout", 5)

        try:
            print(f"{Fore.CYAN}[SETTINGS] Loading memory from {url}...{Style.RESET_ALL}")
            response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                global memory
                loaded_memory = response.json()
                if isinstance(loaded_memory, dict) and "conversation" in loaded_memory:
                    # Validate structure before assigning
                    memory = {
                        "conversation": loaded_memory.get("conversation", []),
                        "observations": loaded_memory.get("observations", []),
                        "files": loaded_memory.get("files", {}),
                        "browser_state": loaded_memory.get("browser_state", None)
                    }
                    # Rebuild system prompt with potentially loaded file list/browser state
                    self.system_prompt = self._build_system_prompt()
                    print(f"{Fore.GREEN} Memory loaded successfully.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}! Invalid memory format received. Starting fresh.{Style.RESET_ALL}")
                    memory = {"conversation": [], "observations": [], "files": {}, "browser_state": None} # Reset memory
            elif response.status_code == 404:
                print(f"{Fore.YELLOW}! No existing memory found on server. Starting fresh.{Style.RESET_ALL}")
                memory = {"conversation": [], "observations": [], "files": {}, "browser_state": None} # Reset memory
            else:
                print(f"{Fore.YELLOW}! Failed to load memory (Status {response.status_code}): {response.text}. Starting fresh.{Style.RESET_ALL}")
                memory = {"conversation": [], "observations": [], "files": {}, "browser_state": None} # Reset memory
        except requests.exceptions.Timeout:
            print(f"{Fore.YELLOW}! Memory server connection timed out after {timeout}s. Using in-memory storage only.{Style.RESET_ALL}")
            if not CONFIG.get("auto_retry_memory_connection", False):
                print(f"{Fore.YELLOW}! Disabling memory server for this session.{Style.RESET_ALL}")
                CONFIG["memory_server_enabled"] = False
        except requests.exceptions.ConnectionError as e:
            print(f"{Fore.YELLOW}! Could not connect to memory server: {e}.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}! Make sure the memory server is running at {CONFIG['memory_server_url']}.{Style.RESET_ALL}")
            if not CONFIG.get("auto_retry_memory_connection", False):
                print(f"{Fore.YELLOW}! Disabling memory server for this session.{Style.RESET_ALL}")
                CONFIG["memory_server_enabled"] = False
        except requests.exceptions.RequestException as e:
            print(f"{Fore.YELLOW}! Memory server request failed: {e}. Using in-memory storage only.{Style.RESET_ALL}")
            if not CONFIG.get("auto_retry_memory_connection", False):
                CONFIG["memory_server_enabled"] = False
        except json.JSONDecodeError:
            print(f"{Fore.YELLOW}! Failed to decode memory JSON from server. Starting fresh.{Style.RESET_ALL}")
            memory = {"conversation": [], "observations": [], "files": {}, "browser_state": None} # Reset memory
        except Exception as e:
            print(f"{Fore.YELLOW}! An unexpected error occurred during memory load: {e}. Using in-memory storage only.{Style.RESET_ALL}")
            # Don't disable server for unexpected errors during load unless critical
            # CONFIG["memory_server_enabled"] = False
        # Ensure system prompt is built even if load fails/is disabled
        self.system_prompt = self._build_system_prompt()


    def save_memory(self):
        """Save agent memory to server"""
        if not CONFIG.get("memory_server_enabled"):
            return

        url = f"{CONFIG['memory_server_url']}/memory/{CONFIG['memory_id']}"
        try:
            # Update system prompt before saving in case state changed (though it's not saved)
            # self.system_prompt = self._build_system_prompt() # Prompt is generated, not saved
            # Optionally prune memory before saving if it gets too large
            # pruned_memory = self._prune_memory(memory)
            # response = requests.post(url, json=pruned_memory, timeout=5)
            response = requests.post(url, json=memory, timeout=CONFIG.get("memory_server_timeout", 5))
            if response.status_code not in (200, 201):
                print(f"{Fore.YELLOW}! Failed to save memory (Status {response.status_code}): {response.text}{Style.RESET_ALL}")
        except requests.exceptions.Timeout:
            print(f"{Fore.YELLOW}! Memory server connection timed out during save.{Style.RESET_ALL}")
        except requests.exceptions.RequestException as e:
            print(f"{Fore.YELLOW}! Could not connect to memory server during save: {e}{Style.RESET_ALL}")
        except Exception as e:
             print(f"{Fore.YELLOW}! An unexpected error occurred during memory save: {e}{Style.RESET_ALL}")


    def run(self):
        """Main CLI loop"""
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[ROBOT] AgentCLI - Autonomous AI Assistant {Style.RESET_ALL}")
        print(f"{Fore.CYAN}Model: {self.model_name}{Style.RESET_ALL}") # Use self.model_name
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"Type your request, or '{Fore.GREEN}help{Style.RESET_ALL}' for commands, '{Fore.RED}exit{Style.RESET_ALL}' to quit.")

        while self.running:
            try:
                # Update system prompt before each user input to reflect latest state
                self.system_prompt = self._build_system_prompt()
                try:
                    user_input = input(f"\n{Fore.GREEN}You{Style.RESET_ALL} > ")
                except UnicodeDecodeError:
                    # Fallback for potential encoding issues on some terminals
                    user_input = input("\nYou > ").encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)


                if not user_input:
                    continue

                user_input_lower = user_input.lower().strip()
                if user_input_lower == 'exit':
                    self.running = False
                    print(f"\n{Fore.CYAN}Saving final state and exiting...{Style.RESET_ALL}")
                    self.save_memory()
                    break

                if user_input_lower == 'help':
                    self.show_help()
                    continue

                if user_input_lower == 'clear':
                    self.clear_memory()
                    continue

                # If it's not a command, process as autonomous request
                self.process_request_autonomous(user_input)

            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}Interrupt received. Saving state and exiting...{Style.RESET_ALL}")
                self.save_memory()
                self.running = False
            except EOFError:
                 print(f"\n\n{Fore.YELLOW}EOF received (likely Ctrl+D). Saving state and exiting...{Style.RESET_ALL}")
                 self.save_memory()
                 self.running = False
            except Exception as e:
                print(f"\n{Fore.RED}{Style.BRIGHT}An unexpected error occurred in the main loop:{Style.RESET_ALL}")
                print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Attempting to save state...{Style.RESET_ALL}")
                self.save_memory()

    def show_help(self):
        """Display help information"""
        # Update system prompt before showing help to reflect current state
        self.system_prompt = self._build_system_prompt()
        help_text = f"""
{Fore.CYAN}AgentCLI Help:{Style.RESET_ALL}

Enter a request for the agent to fulfill (e.g., "Create a python script to list files").
The agent will operate autonomously, thinking, planning, and executing actions step-by-step.

{Fore.CYAN}Commands:{Style.RESET_ALL}
  {Fore.GREEN}help{Style.RESET_ALL}          Show this help message.
  {Fore.GREEN}exit{Style.RESET_ALL}          Save memory (if enabled) and exit the application.
  {Fore.GREEN}clear{Style.RESET_ALL}         Clear the current session's memory (conversation, observations, files).

{Fore.CYAN}Available Actions (as defined in current System Prompt):{Style.RESET_ALL}
{self._extract_actions_from_prompt()}

{Fore.CYAN}Configuration ({Fore.YELLOW}Read Only{Style.RESET_ALL}):
  Model: {self.model_name}
  Max Iterations/Request: {CONFIG['max_iterations_per_request']}
  Memory Server: {'Enabled' if CONFIG['memory_server_enabled'] else 'Disabled'} ({CONFIG['memory_server_url']})
  Puppeteer Server: {'Enabled' if CONFIG['puppeteer_server_enabled'] else 'Disabled'} ({CONFIG['puppeteer_server_url']})
  Confirm Risky Actions: {CONFIG['confirm_risky_actions']}
        """
        print(help_text)

    def _extract_actions_from_prompt(self) -> str:
         """Extracts the action list from the system prompt for display."""
         try:
              action_section = re.search(r"Available Actions.*?):(.*?)Your Response Format:", self.system_prompt, re.DOTALL | re.IGNORECASE)
              if action_section:
                   actions = action_section.group(1).strip()
                   # Indent each line for better display
                   return "\n".join([f"  {line.strip()}" for line in actions.splitlines() if line.strip() and re.match(r'^\d+\.', line.strip())])
              return "  (Could not extract actions from prompt)"
         except Exception:
              return "  (Error extracting actions)"


    def clear_memory(self):
        """Clear the current session memory"""
        global memory
        print(f"{Fore.YELLOW}Clearing session memory (conversation, observations, files, browser state)...{Style.RESET_ALL}")
        memory["conversation"] = []
        memory["observations"] = []
        memory["files"] = {}
        memory["browser_state"] = None
        # Update system prompt after clearing
        self.system_prompt = self._build_system_prompt()
        print(f"{Fore.GREEN} Session memory cleared.{Style.RESET_ALL}")
        # Optionally save the cleared state immediately
        self.save_memory()


    # *** Modified to use genai.Content/Part ***
    def prepare_context_for_api(self, user_goal: str) -> List[dict]:
        """Prepare the context in a format compatible with Gemini API."""
        # Note: System prompt is handled separately in GenerationConfig.
        # The 'contents' list should be the conversation history.

        api_contents = [] # Simple list of role/content dictionaries for compatibility
        conv_idx = 0
        obs_idx = 0

        last_role = None

        # Interleave conversation and observations chronologically
        while conv_idx < len(memory["conversation"]) or obs_idx < len(memory["observations"]):
            # Determine which comes next based on assumption: observation follows assistant message
            conv_entry = memory["conversation"][conv_idx] if conv_idx < len(memory["conversation"]) else None
            obs_entry = memory["observations"][obs_idx] if obs_idx < len(memory["observations"]) else None

            # Prioritize conversation entry unless an observation should follow the last assistant message
            use_conv = True
            # Observation should come after a 'model' role message was added
            if obs_entry is not None and last_role == "model":
                 use_conv = False

            if use_conv and conv_entry is not None:
                role = conv_entry["role"]
                # Map roles: 'assistant' -> 'model'
                api_role = "model" if role == "assistant" else "user"
                content_text = conv_entry.get("content", "")
                if not isinstance(content_text, str): content_text = str(content_text)

                # Skip adding empty content unless it's essential for structure
                if content_text.strip():
                     # Create a simple role/content dictionary compatible with GenerativeModel.generate_content
                     api_contents.append({"role": api_role, "parts": [{"text": content_text}]})
                     last_role = api_role # Track the last role added
                conv_idx += 1
            elif not use_conv and obs_entry is not None:
                 # Observations are treated as input from the 'user' side (environment feedback)
                 action_str = obs_entry.get('action', 'Unknown Action')
                 result_dict = obs_entry.get('result', {})
                 result_obj = ActionResult(**result_dict) # Recreate object for consistent string representation
                 result_str = str(result_obj)
                 # Truncate observation text if it's very long
                 max_obs_len = 1500 # Example limit
                 if len(result_str) > max_obs_len:
                      result_str = result_str[:max_obs_len] + f"... (truncated, full length: {len(result_str)})"

                 obs_text = f"--- Observation for: {action_str[:100]}{'...' if len(action_str) > 100 else ''} ---\n{result_str}\n--- End Observation ---"
                 # Create a simple role/content dictionary compatible with GenerativeModel.generate_content
                 api_contents.append({"role": "user", "parts": [{"text": obs_text}]})
                 last_role = "user" # Observation acts as user input
                 obs_idx += 1
            else:
                # Should not happen if logic is correct, but break loop if it does
                break

        # Add the current user request as the final user message
        api_contents.append({"role": "user", "parts": [{"text": f"Current User Request: {user_goal}"}]})

        return api_contents


    def parse_gemini_response(self, response_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parses thinking, action, and final_response tags from the LLM output."""
        # This function remains largely the same, as it parses the *text content*
        # generated by the model, which should still follow the requested format.
        thinking = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
        action = re.search(r"<action>(.*?)</action>", response_text, re.DOTALL)
        final_response = re.search(r"<final_response>(.*?)</final_response>", response_text, re.DOTALL)

        thinking_text = thinking.group(1).strip() if thinking else None
        action_text = action.group(1).strip() if action else None
        final_response_text = final_response.group(1).strip() if final_response else None

        # Add more robust handling for missing tags or unexpected formats
        if action_text and final_response_text:
             print(f"{Fore.YELLOW}Warning: LLM response contained both <action> and <final_response>. Prioritizing <action>.{Style.RESET_ALL}")
             final_response_text = None # Ignore final_response if action is present

        if not action_text and not final_response_text:
            if thinking_text:
                # If only thinking is present, maybe treat it as the final response? Or log a warning.
                print(f"{Fore.YELLOW}Warning: LLM response contained <thinking> but no <action> or <final_response>. Treating as final response.{Style.RESET_ALL}")
                final_response_text = thinking_text # Assume thinking is the final word
            else:
                # If none of the expected tags are found, it's an error
                print(f"{Fore.RED}Error: LLM response did not contain valid tags (<thinking>, <action>, <final_response>). Treating as error.{Style.RESET_ALL}")
                print(f"Raw Response (first 500 chars): {response_text[:500]}")
                # Synthesize an error message for the agent's flow
                thinking_text = "Agent Error: Received unparseable response."
                final_response_text = f"I received an unexpected response format from the AI model and cannot proceed. Response snippet: {response_text[:200]}"

        return thinking_text, action_text, final_response_text


    def _create_cache_key(self, api_contents: List[dict]) -> str:
        """Create a cache key based on the conversation content."""
        # Create a simplified representation of the conversation for hashing
        key_parts = []
        for content in api_contents:
            if isinstance(content, dict):
                role = content.get('role', '')
                parts = content.get('parts', [])
                text_parts = []
                for part in parts:
                    if isinstance(part, dict) and 'text' in part:
                        text_parts.append(part['text'])
                key_parts.append(f"{role}:{''.join(text_parts)[:100]}")

        # Join and hash the key parts
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def query_gemini(self, api_contents: List[dict]) -> str:
        """Query the Gemini model using the GenerativeModel class and return the model's text response."""
        print(f"{Fore.CYAN}[BRAIN] Querying Gemini ({self.model_name}) using GenerativeModel...{Style.RESET_ALL}")

        # Check if response caching is enabled and if we have a cached response
        if CONFIG.get("enable_response_cache", False):
            # Create a cache key based on the conversation content
            cache_key = self._create_cache_key(api_contents)
            if cache_key in response_cache:
                cache_entry = response_cache[cache_key]
                # Check if the cache entry is still valid
                if time.time() - cache_entry["timestamp"] < CONFIG.get("cache_expiry_seconds", 3600):
                    print(f"{Fore.GREEN}[CACHE] Using cached response from {datetime.fromtimestamp(cache_entry['timestamp']).strftime('%H:%M:%S')}{Style.RESET_ALL}")
                    return cache_entry["response"]
                else:
                    # Remove expired cache entry
                    del response_cache[cache_key]
                    print(f"{Fore.YELLOW}[CACHE] Cached response expired, generating new response{Style.RESET_ALL}")

        # Check if GenerativeModel exists before trying to use it
        if not hasattr(genai, 'GenerativeModel'):
            raise AttributeError("genai module is missing GenerativeModel attribute. Cannot generate content.")

        # Safety settings using standard format
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

        # Generation Config with settings from CONFIG
        generation_config = genai.GenerationConfig(
            temperature=CONFIG.get("model_temperature", 0.4),
            max_output_tokens=CONFIG.get("model_max_output_tokens", 8192),
        )

        # --- Use the GenerativeModel directly ---
        full_response_text = ""
        stream = None # Initialize stream variable
        try:
            # Create a GenerativeModel instance
            model = genai.GenerativeModel(model_name=self.model_name)

            # Generate content with streaming
            # Prepend system prompt as a user message at the start of conversation
            system_contents = [{"role": "user", "parts": [{"text": f"[SYSTEM INSTRUCTION] {self.system_prompt}"}]}]
            # Add system prompt message before actual content
            full_contents = system_contents + api_contents

            # Use progress indicator if enabled
            if CONFIG.get("show_progress_bars", True):
                print(f"{Fore.CYAN}[HOURGLASS] Receiving response stream...{Style.RESET_ALL}", end='', flush=True)
                stream = model.generate_content(
                    contents=full_contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True
                )
            else:
                # If progress bars are disabled, still use streaming for better UX
                stream = model.generate_content(
                    contents=full_contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True
                )

            for chunk in stream:
                # Extract text safely with better error handling
                chunk_text = ""
                try:
                    # First try to access text property directly
                    if hasattr(chunk, 'text'):
                        try:
                            chunk_text = chunk.text
                        except (ValueError, AttributeError) as e:
                            # Handle the specific error when text accessor fails
                            print(f"\n{Fore.YELLOW}Warning: Could not access chunk.text: {e}{Style.RESET_ALL}")

                    # Fallback to parts if text isn't available or failed
                    if not chunk_text and hasattr(chunk, 'parts') and chunk.parts:
                        # Safely extract text from each part
                        part_texts = []
                        for part in chunk.parts:
                            try:
                                if hasattr(part, 'text'):
                                    part_texts.append(part.text)
                            except (ValueError, AttributeError):
                                # Skip parts that don't have valid text
                                pass
                        chunk_text = "".join(part_texts)

                    # Final fallback if we have candidates with text
                    if not chunk_text and hasattr(chunk, 'candidates') and chunk.candidates:
                        # Try to extract text from candidates
                        for candidate in chunk.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        try:
                                            if hasattr(part, 'text'):
                                                chunk_text += part.text
                                        except (ValueError, AttributeError):
                                            pass
                except Exception as chunk_e:
                    # Log the error but continue processing
                    print(f"\n{Fore.YELLOW}Warning: Error processing chunk: {chunk_e}{Style.RESET_ALL}")

                if chunk_text:
                     full_response_text += chunk_text
                     # Simple progress indicator
                     print(".", end="", flush=True)

                # Check for finish reason on each chunk if available
                if hasattr(chunk, 'finish_reason') and chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                    if finish_reason != 0 and finish_reason != 'STOP':
                        print(f"\n{Fore.YELLOW}Warning: Chunk has non-standard finish reason: {finish_reason}{Style.RESET_ALL}")

            print(f" {Fore.GREEN}Done.{Style.RESET_ALL}") # Indicate streaming finished

            # --- Post-stream checks ---
            # Check for errors in the final response
            if not full_response_text.strip():
                print(f"\n{Fore.YELLOW}Warning: Received empty response from the API.{Style.RESET_ALL}")
                # Try to find error information if available
                finish_reason = None
                if hasattr(stream, 'finish_reason'):
                    finish_reason = stream.finish_reason
                elif hasattr(stream, 'candidates') and stream.candidates and hasattr(stream.candidates[0], 'finish_reason'):
                    finish_reason = stream.candidates[0].finish_reason

                if finish_reason is not None and finish_reason != 0 and finish_reason != 'STOP':
                    print(f"{Fore.RED}Error: Generation stopped with finish reason: {finish_reason}{Style.RESET_ALL}")
                    return f"<thinking>Generation error with finish reason: {finish_reason}</thinking><final_response>I encountered an error with the API that prevented me from generating a proper response.</final_response>"

            # Check prompt feedback if available on the stream object after iteration
            prompt_feedback = getattr(stream, 'prompt_feedback', None)
            if prompt_feedback and getattr(prompt_feedback, 'block_reason', None):
                 block_reason = getattr(prompt_feedback, 'block_reason', 'Unknown')
                 # SafetyRating might be under feedback or candidates depending on API version
                 block_message_parts = []
                 if hasattr(prompt_feedback, 'safety_ratings'):
                     for rating in prompt_feedback.safety_ratings:
                          if rating.blocked:
                               block_message_parts.append(f"{rating.category.name}: Blocked")
                 block_message = " - ".join(block_message_parts) if block_message_parts else "(No detailed message)"

                 print(f"{Fore.RED}Error: Gemini response blocked after streaming. Reason: {block_reason} {block_message}{Style.RESET_ALL}")
                 thinking = f"Response blocked by safety filter. Reason: {block_reason}"
                 final_response = "I cannot proceed due to content safety filters blocking the response generation. Please rephrase or simplify the request."
                 return f"<thinking>{thinking}</thinking><final_response>{final_response}</final_response>" # Return structured error

            # Check finish reason from candidates if available
            final_candidates = getattr(stream, 'candidates', None)
            if final_candidates:
                 # Find the finish reason from the first candidate
                 finish_reason_obj = getattr(final_candidates[0], 'finish_reason', None)
                 finish_reason = getattr(finish_reason_obj, 'name', 'UNKNOWN') if finish_reason_obj else 'UNKNOWN'
                 if finish_reason not in ['STOP', 'UNKNOWN', 'FINISH_REASON_UNSPECIFIED']: # Check against known 'good' reasons
                      print(f"{Fore.YELLOW}Warning: Gemini response generation may have stopped prematurely after streaming. Reason: {finish_reason}{Style.RESET_ALL}")
                      # Optionally create structured error, or just return the potentially truncated text

            # If we got here and full_response_text is empty, something went wrong
            if not full_response_text and not (prompt_feedback and getattr(prompt_feedback, 'block_reason', None)): # Avoid double error if blocked
                print(f"{Fore.RED}Error: Received empty response from Gemini model after streaming, and not blocked.{Style.RESET_ALL}")
                return "<thinking>Agent Error: Received empty response.</thinking><final_response>I received an empty response from the AI model and cannot proceed.</final_response>"


            return full_response_text.strip() # Return the assembled text

        # --- Exception Handling ---
        # Specific exceptions from google.api_core or genai library might be useful here
        except AttributeError as ae:
             # Catch attribute errors specifically, often indicates API mismatch
             print(f"{Fore.RED}Error: Attribute error during Gemini API call: {ae}{Style.RESET_ALL}")
             print(f"{Fore.YELLOW}This might indicate an issue with the google-generativeai library version or the API structure.{Style.RESET_ALL}")
             print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
             return f"<thinking>Agent Error: Internal API structure mismatch: {ae}</thinking><final_response>An internal error occurred due to an API incompatibility. Please check library versions.</final_response>"
        except Exception as e:
            # Broad exception catch remains
            print(f"{Fore.RED}Error during Gemini API call: {e}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            # Check if it's a known genai exception type if possible
            # Example: try checking for google.api_core.exceptions.PermissionDenied, etc.

            # Return a structured error message that parse_gemini_response can handle
            return f"<thinking>Agent Error: Exception contacting Gemini API: {e}</thinking><final_response>An error occurred while communicating with the AI model. Please check the logs and try again.</final_response>"


    def execute_parsed_action(self, action_str: str) -> ActionResult:
        """Executes a single, parsed action string and returns the result."""
        # Truncate long action strings for display
        display_action = action_str[:200] + '...' if len(action_str) > 200 else action_str
        print(f"{Fore.CYAN}[ACTION] Executing Action: {Fore.YELLOW}{display_action}{Style.RESET_ALL}")

        match = re.match(r'^\s*([A-Z_]+)\((.*)\)\s*$', action_str, re.DOTALL)
        action_name = None # Initialize
        params_str = ""    # Initialize
        if not match:
            # Handle case where action might be just the name (no params)
            simple_match = re.match(r'^\s*([A-Z_]+)\s*$', action_str)
            if simple_match:
                 action_name = simple_match.group(1).strip()
                 params_str = ""
                 print(f"{Fore.YELLOW}Warning: Action '{action_name}' called without parentheses/parameters. Assuming no parameters needed.{Style.RESET_ALL}")
            else:
                 # If it doesn't match action(...) or just ACTION_NAME, it's invalid
                 error_msg = f"Invalid action format: Action should be ACTION_NAME(parameter='value', ...) or ACTION_NAME. Found: {action_str}"
                 print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
                 return ActionResult(status='error', error=error_msg)
        else:
            action_name = match.group(1).strip()
            params_str = match.group(2).strip()

        # --- Confirmation for risky actions ---
        if CONFIG["confirm_risky_actions"] and action_name in ['DELETE_FILE', 'EXECUTE_COMMAND']:
             try:
                  confirm = input(f"{Fore.YELLOW}Confirm execution of risky action '{action_name}'? (y/N): {Style.RESET_ALL}").lower().strip()
                  if confirm != 'y':
                       print(f"{Fore.RED}Execution cancelled by user.{Style.RESET_ALL}")
                       return ActionResult(status='error', error='Execution cancelled by user.')
             except EOFError: # Handle case where input stream ends during confirmation
                  print(f"{Fore.RED}Input stream ended during confirmation. Cancelling action.{Style.RESET_ALL}")
                  return ActionResult(status='error', error='Execution cancelled due to EOF during confirmation.')


        # --- Parameter Parsing and Execution ---
        try:
            # Use the improved parser with escape handling
            parsed_params = _parse_key_value_parameters(params_str) if params_str else {}
            if not isinstance(parsed_params, dict):
                 # This case shouldn't happen if _parse_key_value_parameters always returns a dict or raises
                 print(f"{Fore.RED}Error: Parameter parsing returned non-dict type: {type(parsed_params)}{Style.RESET_ALL}")
                 return ActionResult(status='error', error="Internal error: Failed to parse action parameters correctly.")

            # --- Action Execution Logic ---
            if action_name in ['CREATE_FILE', 'EDIT_FILE']:
                filename = parsed_params.get('filename')
                content = parsed_params.get('content') # Used by CREATE_FILE
                new_content = parsed_params.get('new_content') # Used by EDIT_FILE

                if not filename: return ActionResult(status='error', error=f"Missing 'filename' parameter for {action_name}")

                if action_name == 'CREATE_FILE':
                    # Content can be None for CREATE_FILE, default to empty string
                    create_content = content if content is not None else ""
                    # Note: create_content will now have unescaped sequences like \n converted to newlines
                    return self.create_file(filename, create_content)
                elif action_name == 'EDIT_FILE':
                    # new_content is required for EDIT_FILE
                    if new_content is None:
                        return ActionResult(status='error', error=f"Missing 'new_content' parameter for {action_name}")
                    # Note: new_content will now have unescaped sequences like \n converted to newlines
                    return self.edit_file(filename, new_content)

            elif action_name in ['READ_FILE', 'DELETE_FILE']:
                filename = parsed_params.get('filename')
                if not filename: return ActionResult(status='error', error=f"Missing 'filename' parameter for {action_name}")
                if action_name == 'READ_FILE': return self.read_file(filename)
                elif action_name == 'DELETE_FILE': return self.delete_file(filename)

            elif action_name == 'EXECUTE_COMMAND':
                command = parsed_params.get('command')
                if not command: return ActionResult(status='error', error=f"Missing 'command' parameter for {action_name}")
                return self.execute_command(command)

            elif action_name in ['BROWSE_WEB', 'SEARCH_WEB']:
                url = parsed_params.get('url')
                query = parsed_params.get('query')
                if action_name == 'BROWSE_WEB':
                    if not url: return ActionResult(status='error', error=f"Missing 'url' parameter for {action_name}")
                    return self.browse_web(url)
                elif action_name == 'SEARCH_WEB':
                    if not query: return ActionResult(status='error', error=f"Missing 'query' parameter for {action_name}")
                    return self.search_web(query)

            elif action_name == 'SCREENSHOT_WEB':
                # Selector is optional
                selector = parsed_params.get('selector')
                return self.screenshot_web(selector) # Pass None if key missing

            elif action_name == 'LIST_FILES':
                directory = parsed_params.get('directory', '.')
                pattern = parsed_params.get('pattern', '*')
                return self.list_files(directory, pattern)

            elif action_name == 'FIND_IN_FILES':
                pattern = parsed_params.get('pattern')
                if not pattern: return ActionResult(status='error', error=f"Missing 'pattern' parameter for {action_name}")
                directory = parsed_params.get('directory', '.')
                file_pattern = parsed_params.get('file_pattern', '*')
                return self.find_in_files(pattern, directory, file_pattern)

            elif action_name == 'EXECUTE_PYTHON' and CONFIG.get("enable_code_execution", False):
                code = parsed_params.get('code')
                if not code: return ActionResult(status='error', error=f"Missing 'code' parameter for {action_name}")
                return self.execute_python(code)

            elif action_name == 'EXECUTE_JAVASCRIPT' and CONFIG.get("enable_code_execution", False):
                code = parsed_params.get('code')
                if not code: return ActionResult(status='error', error=f"Missing 'code' parameter for {action_name}")
                return self.execute_javascript(code)

            elif action_name == 'DIFF_FILES' and CONFIG.get("enable_file_diff", False):
                file1 = parsed_params.get('file1')
                file2 = parsed_params.get('file2')
                if not file1: return ActionResult(status='error', error=f"Missing 'file1' parameter for {action_name}")
                if not file2: return ActionResult(status='error', error=f"Missing 'file2' parameter for {action_name}")
                return self.diff_files(file1, file2)

            elif action_name == 'COPY_TO_CLIPBOARD' and CONFIG.get("enable_clipboard", False):
                text = parsed_params.get('text')
                if not text: return ActionResult(status='error', error=f"Missing 'text' parameter for {action_name}")
                return self.copy_to_clipboard(text)

            elif action_name == 'GET_CLIPBOARD' and CONFIG.get("enable_clipboard", False):
                return self.get_clipboard()

            elif action_name == 'DOWNLOAD_FILE':
                url = parsed_params.get('url')
                filename = parsed_params.get('filename')
                if not url: return ActionResult(status='error', error=f"Missing 'url' parameter for {action_name}")
                if not filename: return ActionResult(status='error', error=f"Missing 'filename' parameter for {action_name}")
                return self.download_file(url, filename)

            elif action_name == 'SUMMARIZE_TEXT':
                text = parsed_params.get('text')
                if not text: return ActionResult(status='error', error=f"Missing 'text' parameter for {action_name}")
                return self.summarize_text(text)

            else:
                return ActionResult(status='error', error=f"Unknown action: {action_name}")

        except Exception as e:
            print(f"{Fore.RED}Critical Error during action execution ({action_name or 'Unknown Action'}): {e}{Style.RESET_ALL}")
            print(f"{Fore.RED}Traceback: {traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=f"Unexpected exception during {action_name or 'Unknown Action'}: {e}")


    def process_request_autonomous(self, user_input: str):
        """Processes the user request using an autonomous think-act-observe loop."""
        print(f"\n{Fore.MAGENTA}--- Starting Autonomous Execution ---{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Goal: {user_input}{Style.RESET_ALL}")

        # Add user input to conversation history *before* the first API call for this request
        memory["conversation"].append({"role": "user", "content": user_input})

        iteration = 0
        while iteration < CONFIG["max_iterations_per_request"]:
            iteration += 1
            print(f"\n{Fore.BLUE}--- Iteration {iteration}/{CONFIG['max_iterations_per_request']} ---{Style.RESET_ALL}")

            action_name_for_prompt = "None" # For updating prompt after action
            action_succeeded = False

            try:
                # Update system prompt to reflect latest state before preparing context
                self.system_prompt = self._build_system_prompt()

                # Prepare context in the format required by the new API call
                api_contents = self.prepare_context_for_api(user_input) # Pass the original goal

                # Query Gemini using the new structured method
                raw_response = self.query_gemini(api_contents)

                # Add raw response to conversation history (as 'assistant'/'model')
                memory["conversation"].append({"role": "assistant", "content": raw_response})

                # Parse the response (which should contain <thinking>, <action>/<final_response>)
                thinking, action_str, final_response_str = self.parse_gemini_response(raw_response)

                if thinking: print(f"{Fore.YELLOW}[THINKING] Agent Thinking:\n{thinking}{Style.RESET_ALL}")
                else: print(f"{Fore.YELLOW}[THINKING] Agent Thinking: (No thinking provided){Style.RESET_ALL}")

                if action_str:
                    # Extract action name before execution for later prompt update check
                    action_match = re.match(r'^\s*([A-Z_]+)', action_str)
                    action_name_for_prompt = action_match.group(1) if action_match else "Unknown"

                    action_result = self.execute_parsed_action(action_str)
                    action_succeeded = action_result.status == 'success'

                    # Add observation *before* the next loop/API call
                    memory["observations"].append({"action": action_str, "result": action_result.to_dict()})

                    if action_succeeded:
                        print(f"{Fore.GREEN} Action Result:\n{action_result}{Style.RESET_ALL}")
                        # Prompt is updated at the START of the next loop now
                        # if action_name_for_prompt in ['CREATE_FILE', 'EDIT_FILE', 'DELETE_FILE', 'BROWSE_WEB', 'SCREENSHOT_WEB']: # Include screenshot saving
                        #      self.system_prompt = self._build_system_prompt() # Update prompt immediately if needed?
                    else:
                        print(f"{Fore.RED}[X] Action Error:\n{action_result}{Style.RESET_ALL}")
                        # The error observation will be sent in the next iteration

                    self.save_memory() # Save after each action/observation cycle

                elif final_response_str:
                    print(f"\n{Fore.GREEN}[FINISH] Agent Finished:{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}{final_response_str}{Style.RESET_ALL}")
                    # No more actions, break the loop
                    break # Exit the while loop
                else:
                    # This case is handled inside parse_gemini_response now, which should return
                    # a final_response_str indicating the error. If we still get here, it's unexpected.
                    print(f"{Fore.RED}Error: Agent response parsing failed unexpectedly after API call. Stopping.{Style.RESET_ALL}")
                    # Add a final error message to conversation if needed
                    memory["conversation"].append({"role": "assistant", "content": "<final_response>Agent Error: Internal loop error after response parsing.</final_response>"})
                    break # Exit the while loop

            except Exception as loop_error:
                 print(f"\n{Fore.RED}{Style.BRIGHT}An error occurred within the autonomous loop (Iteration {iteration}):{Style.RESET_ALL}")
                 print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
                 # Add error observation if an action was attempted? Maybe too complex.
                 # Synthesize a final response indicating the loop error.
                 error_final_response = f"<thinking>An unexpected error occurred during processing: {loop_error}</thinking><final_response>I encountered an internal error ({type(loop_error).__name__}) during execution and cannot continue with the current task. Please check the logs.</final_response>"
                 memory["conversation"].append({"role": "assistant", "content": error_final_response})
                 print(f"{Fore.RED}Loop terminated due to error.{Style.RESET_ALL}")
                 break # Exit the while loop


        # After the loop finishes (either by break, completion, or max iterations)
        if iteration >= CONFIG["max_iterations_per_request"]:
            print(f"\n{Fore.RED}[FINISH] Agent Stopped: Maximum iterations ({CONFIG['max_iterations_per_request']}) reached.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Task may be incomplete. You can ask the agent to continue, provide refinement, or start a new task.{Style.RESET_ALL}")
            # Add a final message indicating max iterations reached
            if not memory["conversation"][-1]["content"].strip().endswith("</final_response>"): # Avoid adding if last msg was already final
                memory["conversation"].append({"role": "assistant", "content": "<thinking>Reached maximum iteration limit.</thinking><final_response>I have reached the maximum number of steps allowed for this request. The task may be incomplete. Please let me know if I should continue or if you have further instructions.</final_response>"})

        print(f"{Fore.MAGENTA}--- Autonomous Execution Finished ---{Style.RESET_ALL}")
        self.save_memory() # Final save


    # --- Action Implementations (Returning ActionResult) ---
    # These functions remain the same as they implement the actions themselves.

    def _resolve_path(self, filename: str) -> str:
        """Resolves filename to an absolute path within cwd and checks safety."""
        if not filename or not isinstance(filename, str):
             raise ValueError("Invalid filename provided (empty or not a string).")

        # Basic sanitization: remove leading/trailing whitespace
        filename = filename.strip()
        if not filename:
            raise ValueError("Invalid filename provided (empty after stripping whitespace).")

        # Attempt normalization first
        try:
            # Replace backslashes with forward slashes for consistency before normpath
            # (Helps mainly on Windows where mixed separators can occur)
            normalized_filename = os.path.normpath(filename.replace('\\', '/'))
        except Exception as e:
            raise ValueError(f"Invalid filename path (normalization failed): {e}")


        # Disallow absolute paths explicitly based on os.path.isabs
        # This check should work reliably across OSes after normpath
        if os.path.isabs(normalized_filename):
            raise ValueError("Invalid filename path (absolute paths are not allowed).")

        # Disallow paths starting with '..' AFTER normalization
        # This catches direct attempts like "../file" or "subdir/../../file"
        # os.path.normpath resolves "foo/../bar" to "bar" on Unix-like
        # and "foo\\..\\bar" to "bar" on Windows.
        # So, we just need to check if the resulting normalized path starts with '..'
        if normalized_filename.startswith(".."):
             raise ValueError("Invalid filename path (directory traversal '..' detected at the start).")

        # Combine with CWD and get the absolute path
        target_path = os.path.abspath(os.path.join(os.getcwd(), normalized_filename))

        # Final safety check: Ensure the resolved absolute path is still within the CWD tree
        # Need to ensure comparison works correctly even with slightly different path representations
        # (e.g. C:\path vs C:\path\)
        cwd_abspath = os.path.abspath(os.getcwd())
        if not target_path.startswith(cwd_abspath + os.sep) and target_path != cwd_abspath:
             # Check if target_path IS the cwd (shouldn't happen for files)
             # Allow paths directly within cwd
             if os.path.dirname(target_path) == cwd_abspath:
                  pass # It's directly in the CWD, allow it.
             else:
                  print(f"{Fore.YELLOW}Warning: Path resolution resulted outside CWD. CWD: {cwd_abspath}, Target: {target_path}{Style.RESET_ALL}")
                  raise ValueError("Invalid filename path (resolved path is outside the current working directory tree).")

        return target_path

    def _get_relative_path(self, target_path: str) -> str:
         """Gets the path relative to the current working directory."""
         # Ensure target_path is absolute before making it relative to CWD
         if not os.path.isabs(target_path):
              target_path = os.path.abspath(target_path)
         relative = os.path.relpath(target_path, os.getcwd())
         # Avoid returning "." if the path *is* the CWD (though unlikely for file ops)
         return relative if relative != "." else ""


    def create_file(self, filename: str, content: str) -> ActionResult:
        print(f"{Fore.CYAN}[WRITE] Creating file: {filename}{Style.RESET_ALL}")
        # The 'content' received here has already been unescaped by the parser
        try:
            target_path = self._resolve_path(filename)
            relative_path = self._get_relative_path(target_path)
            if not relative_path:
                 raise ValueError("Cannot create file corresponding to the working directory itself.")

            # Check if it already exists and is a directory
            if os.path.isdir(target_path):
                raise ValueError(f"Cannot create file '{relative_path}', a directory with that name already exists.")
            # Check if it already exists and is a file
            if os.path.isfile(target_path):
                # Policy: Error on create if file exists. Use EDIT_FILE.
                raise ValueError(f"Cannot create file '{relative_path}', a file with that name already exists. Use EDIT_FILE to overwrite.")


            dir_path = os.path.dirname(target_path)
            # Only create directories if dir_path is not empty (i.e., not creating in CWD)
            # and if it doesn't already exist.
            if dir_path and not os.path.exists(dir_path):
                 os.makedirs(dir_path, exist_ok=True)
            elif dir_path and not os.path.isdir(dir_path):
                 # If the intended directory path exists but is a file, raise error
                 raise ValueError(f"Cannot create directory '{os.path.dirname(relative_path)}' because a file with that name exists.")


            # Ensure content is a string (already unescaped if originally quoted)
            if not isinstance(content, str):
                print(f"{Fore.YELLOW}Warning: Content for {filename} was not a string ({type(content)}), converting...{Style.RESET_ALL}")
                content = str(content)

            # Special handling for HTML files - ensure valid HTML structure
            # The prompt asks the LLM to generate HTML tags (<p>, <br>) instead of \n.
            # The parser converted any stray \n to actual newlines. We now write the content as is.
            # If the LLM followed instructions, `content` should already be formatted HTML.
            if target_path.lower().endswith(('.html', '.htm')):
                # No extra processing needed here if LLM followed prompt. Write content directly.
                # Example: If LLM sent content="<h1>Hi</h1><p>Line 1<br>Line 2</p>", it gets written as is.
                # Example: If LLM sent content="Line1\\nLine2", parser made it "Line1\nLine2". This will render poorly in HTML.
                # We rely on the LLM prompt instruction for correct HTML generation.
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                # Standard file handling for non-HTML files - write the content (with interpreted newlines)
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            timestamp = time.time()
            # Get size after writing
            file_size = os.path.getsize(target_path)
            memory["files"][relative_path] = {"created": timestamp, "last_modified": timestamp, "size": file_size}
            return ActionResult(status='success', output=f"File '{relative_path}' created successfully ({file_size} bytes).")
        except ValueError as ve: # Catch specific path/validation errors
             error_msg = f"Error creating file '{filename}': {ve}"
             print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
             return ActionResult(status='error', error=error_msg)
        except Exception as e:
            error_msg = f"Error creating file '{filename}': {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def read_file(self, filename: str) -> ActionResult:
        print(f"{Fore.CYAN}[FILE] Reading file: {filename}{Style.RESET_ALL}")
        try:
            target_path = self._resolve_path(filename)
            relative_path = self._get_relative_path(target_path)
            if not relative_path:
                 raise ValueError("Cannot read the working directory itself as a file.")

            if not os.path.exists(target_path):
                 # If known in memory but not on disk, report discrepancy
                 if relative_path in memory["files"]:
                     del memory["files"][relative_path] # Remove stale entry
                     raise FileNotFoundError(f"File '{relative_path}' known in memory but not found on disk. Removed from memory.")
                 else:
                     raise FileNotFoundError(f"File '{relative_path}' not found on disk.")
            if not os.path.isfile(target_path):
                 raise ValueError(f"'{relative_path}' is not a file (it might be a directory).")

            # Read the file content
            content = None
            file_size = os.path.getsize(target_path)
            binary_warning = ""
            try:
                # Limit read size? Potentially large files could crash agent.
                # max_read_size = 1 * 1024 * 1024 # 1 MB limit?
                # if file_size > max_read_size:
                #     raise ValueError(f"File size ({file_size} bytes) exceeds maximum read limit ({max_read_size} bytes).")

                with open(target_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                 binary_warning = f" Warning: Failed to decode {relative_path} as UTF-8."
                 print(f"{Fore.YELLOW}{binary_warning}{Style.RESET_ALL}")
                 try:
                      # Try reading as binary and providing info
                      with open(target_path, 'rb') as f:
                           # Read only a small part for preview?
                           binary_content_preview = f.read(1024)
                      # Represent binary content safely
                      content = f"<Binary content: {file_size} bytes total. Could not decode as text (UTF-8). Preview (first {len(binary_content_preview)} bytes, Base64 encoded): {base64.b64encode(binary_content_preview).decode()}>"

                 except Exception as bin_e:
                      raise IOError(f"Failed to read '{relative_path}' as text (UTF-8) or binary: {bin_e}")
            except Exception as read_e:
                 # Catch other read errors (e.g., permissions)
                 raise IOError(f"Failed to read '{relative_path}': {read_e}")

            # Update memory with potentially corrected size/timestamp
            memory["files"][relative_path] = {
                **memory["files"].get(relative_path, {}), # Keep existing created time if present
                "last_read": time.time(),
                "size": file_size
            }

            # Truncate content for display/return in output message if very large
            max_return_len = 5000 # Increased limit
            truncated = False
            output_content = content # Start with full content
            if isinstance(content, str) and len(content) > max_return_len:
                 truncated = True
                 output_content = content[:max_return_len] + f"... (truncated, total length {len(content)})"


            print(f"{Fore.YELLOW} Read {file_size} bytes from {relative_path}. Returning {'truncated ' if truncated else ''}content.{binary_warning}{Style.RESET_ALL}")
            # Return the potentially truncated content in the 'data' field for the LLM
            # And a summary message in 'output'
            return ActionResult(status='success',
                                output=f"Read {file_size} bytes from '{relative_path}'.{binary_warning}{' Content truncated.' if truncated else ''}",
                                data=output_content) # Send potentially truncated content back to LLM

        except (FileNotFoundError, ValueError, IOError) as e: # Catch expected errors
            error_msg = f"Error reading file '{filename}': {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            # Clean up memory if file not found but was listed
            if isinstance(e, FileNotFoundError) and 'relative_path' in locals() and relative_path in memory["files"]:
                del memory["files"][relative_path]
            return ActionResult(status='error', error=error_msg)
        except Exception as e: # Catch unexpected errors
            error_msg = f"Unexpected error reading file '{filename}': {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)


    def edit_file(self, filename: str, new_content: str) -> ActionResult:
        print(f"{Fore.CYAN}[EDIT] Editing file: {filename}{Style.RESET_ALL}")
        # The 'new_content' received here has already been unescaped by the parser
        try:
            target_path = self._resolve_path(filename)
            relative_path = self._get_relative_path(target_path)
            if not relative_path:
                 raise ValueError("Cannot edit the working directory itself as a file.")

            action = "created"
            old_content = None
            file_existed = os.path.exists(target_path)
            original_created_time = None

            if file_existed:
                 if not os.path.isfile(target_path):
                     raise ValueError(f"'{relative_path}' exists but is not a file (it's likely a directory). Cannot edit.")
                 action = "overwritten"
                 original_created_time = memory["files"].get(relative_path, {}).get("created")
                 try:
                      # Read existing content for diff summary (optional)
                      with open(target_path, 'r', encoding='utf-8') as f: old_content = f.read()
                 except Exception as read_err:
                      print(f"{Fore.YELLOW}Warning: Could not read existing content of {relative_path} before overwrite: {read_err}{Style.RESET_ALL}")
                      old_content = None # Ensure it's None if read fails
            else:
                 # File does not exist, will be created
                 action = "created"
                 print(f"{Fore.YELLOW}Note: File '{relative_path}' not found, will create new file.{Style.RESET_ALL}")


            # Ensure content is a string (already unescaped)
            if not isinstance(new_content, str):
                print(f"{Fore.YELLOW}Warning: New content for {filename} was not a string ({type(new_content)}), converting...{Style.RESET_ALL}")
                new_content = str(new_content)

            # Ensure directory exists
            dir_path = os.path.dirname(target_path)
            if dir_path and not os.path.exists(dir_path):
                 os.makedirs(dir_path, exist_ok=True)
            elif dir_path and not os.path.isdir(dir_path):
                 raise ValueError(f"Cannot create directory for '{relative_path}' because a file exists at '{os.path.dirname(relative_path)}'.")


            # Write the new content
            # Special handling for HTML files - rely on LLM prompt to generate correct HTML tags.
            if target_path.lower().endswith(('.html', '.htm')):
                # Write the unescaped content directly.
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            else:
                # Standard file handling - write the unescaped content.
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

            timestamp = time.time()
            new_size = os.path.getsize(target_path) # Get size after writing

            # Update memory: add 'created' timestamp if it was just created
            if action == "created":
                memory["files"][relative_path] = {"created": timestamp, "last_modified": timestamp, "size": new_size}
            else: # If overwritten, update modified time and size
                 # Preserve original created time if known, otherwise use current time
                 created_time = original_created_time if original_created_time is not None else timestamp
                 memory["files"][relative_path] = {"created": created_time, "last_modified": timestamp, "size": new_size}


            # Simple diff summary (optional)
            diff_summary = ""
            if old_content is not None and isinstance(old_content, str):
                 # Simple line diff count (approximate)
                 old_lines = set(old_content.splitlines())
                 new_lines = set(new_content.splitlines())
                 added = len(new_lines - old_lines)
                 removed = len(old_lines - new_lines)
                 if added > 0 or removed > 0:
                      diff_summary = f" (approx. +{added}/-{removed} lines changed)"

            output_msg = f"File '{relative_path}' {action} successfully ({new_size} bytes).{diff_summary}"
            print(f"{Fore.GREEN} {output_msg}{Style.RESET_ALL}")
            return ActionResult(status='success', output=output_msg)

        except ValueError as ve: # Catch specific path/validation errors
             error_msg = f"Error editing file '{filename}': {ve}"
             print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
             return ActionResult(status='error', error=error_msg)
        except Exception as e: # Catch other unexpected errors
            error_msg = f"Error editing file '{filename}': {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}") # Log traceback for unexpected
            return ActionResult(status='error', error=error_msg)

    def delete_file(self, filename: str) -> ActionResult:
        print(f"{Fore.CYAN}[TRASH] Deleting file: {filename}{Style.RESET_ALL}")
        try:
            target_path = self._resolve_path(filename)
            relative_path = self._get_relative_path(target_path)
            if not relative_path:
                 raise ValueError("Cannot delete the working directory itself.")


            if not os.path.exists(target_path):
                if relative_path in memory["files"]:
                     del memory["files"][relative_path]
                     return ActionResult(status='success', output=f"File '{relative_path}' was already deleted (or never existed on disk), removed from memory.")
                else: raise FileNotFoundError(f"File '{relative_path}' does not exist.")

            if not os.path.isfile(target_path):
                 # For safety, explicitly disallow deleting directories with this command
                 raise ValueError(f"'{relative_path}' is not a file (it's likely a directory). Cannot delete directories with DELETE_FILE.")

            # Perform the deletion
            os.remove(target_path)

            # Remove from memory if present
            if relative_path in memory["files"]: del memory["files"][relative_path]

            output_msg = f"File '{relative_path}' deleted successfully."
            print(f"{Fore.GREEN} {output_msg}{Style.RESET_ALL}")
            return ActionResult(status='success', output=output_msg)

        except (FileNotFoundError, ValueError, PermissionError) as e: # Catch expected errors
            error_msg = f"Error deleting file '{filename}': {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
             # Clean up memory if file not found but was listed
            if isinstance(e, FileNotFoundError) and 'relative_path' in locals() and relative_path in memory["files"]:
                del memory["files"][relative_path]
            return ActionResult(status='error', error=error_msg)
        except Exception as e: # Catch unexpected errors
            error_msg = f"Unexpected error deleting file '{filename}': {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def execute_command(self, command: str) -> ActionResult:
        if not command or not isinstance(command, str):
            return ActionResult(status='error', error="No command provided or command is not a string.")

        command = command.strip()
        if not command:
             return ActionResult(status='error', error="Command is empty after stripping whitespace.")

        print(f"{Fore.YELLOW}Executing command: {command}{Style.RESET_ALL}")

        # Basic safety check: still fragile, confirmation is key
        # forbidden_patterns = [r'^\s*rm\s+-rf\s+/\s*'] # Example
        # for pattern in forbidden_patterns:
        #      if re.match(pattern, command):
        #           error_msg = f"Execution blocked: Command '{command}' matches a potentially dangerous pattern."
        #           print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        #           return ActionResult(status='error', error=error_msg)

        process = None # Initialize process variable
        try:
            # Using shell=True necessary for complex commands but carries risks.
            # Confirmation prompt adds a layer of safety.
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', cwd=os.getcwd(), # Run in current dir
                # Add creationflags for Windows if needed to hide console window
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == 'Windows' else 0
            )
            # Increased timeout? 120s might be better.
            timeout_seconds = 120
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode # Get exit code *after* communicate

            # Prepare output, truncating long stdout/stderr for the ActionResult message
            max_len = 2000 # Increased max length for returned data
            stdout_brief = stdout
            stderr_brief = stderr
            stdout_truncated = False
            stderr_truncated = False

            if stdout and len(stdout) > max_len:
                stdout_brief = stdout[:max_len] + f'... (truncated, total length {len(stdout)})'
                stdout_truncated = True
            if stderr and len(stderr) > max_len:
                stderr_brief = stderr[:max_len] + f'... (truncated, total length {len(stderr)})'
                stderr_truncated = True


            output_msg = f"Command executed. Exit Code: {exit_code}."
            trunc_msg = ""
            if stdout_truncated and stderr_truncated: trunc_msg = " (stdout/stderr truncated)"
            elif stdout_truncated: trunc_msg = " (stdout truncated)"
            elif stderr_truncated: trunc_msg = " (stderr truncated)"

            print(f"{Fore.GREEN if exit_code == 0 else Fore.YELLOW} {output_msg}{trunc_msg}{Style.RESET_ALL}")

            # Display the potentially truncated versions in the console for brevity
            if stdout_brief: print(f"{Fore.WHITE}Stdout (brief):\n{stdout_brief}{Style.RESET_ALL}")
            if stderr_brief: print(f"{Fore.RED}Stderr (brief):\n{stderr_brief}{Style.RESET_ALL}")

            if exit_code == 0:
                 # Success: Return potentially truncated stdout/stderr in output/error fields for LLM
                 return ActionResult(status='success', output=stdout_brief or output_msg, error=stderr_brief or None)
            else:
                 # Error: Return potentially truncated stdout/stderr
                 # Prioritize stderr for the main error message if it exists
                 error_message = stderr_brief if stderr_brief else f"Command failed with exit code {exit_code} (no stderr)."
                 return ActionResult(status='error', output=stdout_brief or None, error=error_message)

        except subprocess.TimeoutExpired:
             error_msg = f"Command '{command}' timed out after {timeout_seconds} seconds."
             print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
             # Ensure process is killed if it timed out
             if process and process.poll() is None: # Check if process is still running
                 try:
                     process.kill()
                     # Optionally wait a very short time for kill to take effect
                     process.communicate(timeout=1) # Clean up pipes after kill
                 except Exception as kill_e:
                     print(f"{Fore.YELLOW}Warning: Error trying to kill/cleanup timed-out process: {kill_e}{Style.RESET_ALL}")
             return ActionResult(status='error', error=error_msg)
        except FileNotFoundError as fnf_e:
             # Often means the command itself wasn't found in the PATH
             error_msg = f"Error executing command: {fnf_e}. The command '{command.split()[0] if command else ''}' might not be installed or not in the system's PATH."
             print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
             return ActionResult(status='error', error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error executing command '{command}': {e}"
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            # Ensure process is cleaned up if it exists and failed unexpectedly
            if process and process.poll() is None:
                 try: process.kill(); process.communicate(timeout=1)
                 except: pass
            return ActionResult(status='error', error=error_msg)


    # --- Web Operations (using Puppeteer MCP server) ---

    def _call_puppeteer_server(self, endpoint: str, data: Dict, timeout: int = 15) -> ActionResult:
        """Helper function to call the Puppeteer server."""
        if not CONFIG.get("puppeteer_server_enabled"):
            return ActionResult(status='error', error="Puppeteer server is disabled in config.")

        url = f"{CONFIG['puppeteer_server_url']}/{endpoint}"
        response = None # Initialize response variable
        try:
            # Add headers, e.g., Content-Type
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=data, timeout=timeout, headers=headers)

            # Check for non-2xx status codes first
            if not response.ok:
                 # Try to get error details from response body if possible
                 error_detail = f"Raw response: {response.text[:500]}" # Default to raw text
                 try:
                      json_error = response.json()
                      if isinstance(json_error, dict):
                           if 'error' in json_error: error_detail = json_error['error']
                           elif 'message' in json_error: error_detail = json_error['message']
                           else: error_detail = json.dumps(json_error) # Dump whole dict if no known error key
                      elif isinstance(json_error, str): # Handle plain string error response
                           error_detail = json_error
                 except json.JSONDecodeError:
                      pass # Keep raw text if not JSON
                 except Exception as json_parse_e: # Catch other errors during error parsing
                      error_detail += f" (Error parsing error JSON: {json_parse_e})"

                 error_msg = f"Puppeteer server returned error {response.status_code} at '{url}': {error_detail}"
                 print(f"{Fore.RED}[X] {error_msg}{Style.RESET_ALL}")
                 return ActionResult(status='error', error=error_msg)


            # If status code is OK (2xx), then parse the JSON
            result = response.json()

            # Check for application-level errors within the JSON response
            if isinstance(result, dict) and result.get('status') == 'error':
                 puppeteer_error = result.get('error', 'Unknown error message from Puppeteer server.')
                 print(f"{Fore.RED}[X] Puppeteer action failed (API reported error): {puppeteer_error}{Style.RESET_ALL}")
                 # Pass through any other data returned alongside the error
                 return ActionResult(status='error', error=puppeteer_error, data=result)

            # If no 'status'=='error', assume success
            success_message = result.get("message", "Puppeteer action successful.") if isinstance(result, dict) else "Puppeteer action successful (non-dict response)."
            return ActionResult(status='success', output=success_message, data=result)

        except requests.exceptions.Timeout:
            error_msg = f"Timeout connecting to Puppeteer server at {url} after {timeout}s."
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)
        except requests.exceptions.ConnectionError as e:
             error_msg = f"Connection error contacting Puppeteer server at {url}: {e}. Is it running?"
             print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
             return ActionResult(status='error', error=error_msg)
        except requests.exceptions.RequestException as e:
            # Catch other requests errors (like invalid URL, SSL issues)
            error_msg = f"Error during request to Puppeteer server at {url}: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)
        except json.JSONDecodeError:
             raw_resp_text = response.text[:200] if response else "N/A"
             error_msg = f"Failed to decode JSON response from Puppeteer server at {url}. Response text: {raw_resp_text}..."
             print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
             return ActionResult(status='error', error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during Puppeteer call to {url}: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)


    def browse_web(self, url: str) -> ActionResult:
        if not url or not isinstance(url, str):
             return ActionResult(status='error', error="Invalid URL provided.")
        url = url.strip()
        # Simple check: does it look like a URL? Add http:// if missing scheme?
        # Be more robust: check common schemes or assume https
        if not re.match(r'^[a-zA-Z]+://', url):
            print(f"{Fore.YELLOW}Warning: URL '{url}' missing scheme (e.g., http/https), prepending 'https://'.{Style.RESET_ALL}")
            url = f"https://{url}"

        print(f"{Fore.CYAN}[WEB] Browsing to URL: {url}{Style.RESET_ALL}")
        # Increased timeout for browsing as pages can take time to load
        result = self._call_puppeteer_server("browse", {"url": url}, timeout=60)

        if result.status == 'success' and isinstance(result.data, dict):
            # Update browser state in memory
            new_url = result.data.get("url", url) # Use the final URL from puppeteer if available
            title = result.data.get("title", "N/A")
            # Basic sanitization/truncation for title
            title = str(title).strip()[:200] if title else "N/A"

            memory["browser_state"] = {
                "url": new_url,
                "title": title,
                "last_visited": time.time()
            }
            # Update system prompt to reflect new browser state (done at start of next loop)
            # self.system_prompt = self._build_system_prompt()
            print(f"{Fore.GREEN} Successfully loaded page. Title: '{title}'. URL: {new_url}{Style.RESET_ALL}")
            # Modify result output to be more informative for LLM
            result.output = f"Successfully navigated to URL: {new_url}. Page Title: '{title}'."
        elif result.status == 'error':
             # Clear browser state on error? Or keep the old state? Let's clear it for safety.
             memory["browser_state"] = None
             # self.system_prompt = self._build_system_prompt() # Update prompt
             print(f"{Fore.RED}[X] Failed to browse: {result.error}{Style.RESET_ALL}")
        elif result.status == 'success' and not isinstance(result.data, dict):
             # Success but unexpected data format
             print(f"{Fore.YELLOW}! Puppeteer browse call succeeded but returned unexpected data format: {type(result.data)}{Style.RESET_ALL}")
             # Update state minimally
             memory["browser_state"] = { "url": url, "title": "Unknown (Invalid data from server)", "last_visited": time.time()}
             # self.system_prompt = self._build_system_prompt() # Update prompt
             result.output = f"Navigation to {url} seemed successful, but server returned unexpected data."


        return result


    def screenshot_web(self, selector: Optional[str] = None) -> ActionResult:
        if not memory.get("browser_state") or not memory["browser_state"].get("url"):
             return ActionResult(status='error', error="No active browser session with a loaded URL. Use BROWSE_WEB first.")

        current_url = memory["browser_state"]["url"]
        print(f"{Fore.CYAN}[CAMERA] Taking screenshot of current page ({current_url}){(' - Selector: ' + selector) if selector else ''}{Style.RESET_ALL}")

        data = {"format": "png"} # Request PNG format
        if selector and isinstance(selector, str) and selector.strip():
            data["selector"] = selector.strip()

        # Screenshot might also take time
        result = self._call_puppeteer_server("screenshot", data, timeout=45)

        # Process successful screenshot
        if result.status == 'success' and isinstance(result.data, dict) and "data" in result.data:
            # Expect data to be a base64 encoded image string (potentially data URI)
            img_data_uri = result.data["data"]
            if not isinstance(img_data_uri, str) or not img_data_uri:
                error_msg = "Puppeteer server returned empty or invalid image data."
                print(f"{Fore.RED}[X] {error_msg}{Style.RESET_ALL}")
                return ActionResult(status='error', error=error_msg, data=result.data)

            relative_path = None # Initialize
            try:
                screenshots_dir = "screenshots"
                os.makedirs(screenshots_dir, exist_ok=True)

                # Generate a filename based on domain/timestamp
                try:
                    # Extract domain, handle potential errors
                    domain_match = re.search(r'https?://([^/:]+)', current_url) # Include : to stop at port
                    domain = domain_match.group(1) if domain_match else "unknown_domain"
                    # Sanitize domain for filename
                    sanitized_domain = re.sub(r'[^\w\-.]+', '_', domain).strip('_')
                    if not sanitized_domain: sanitized_domain = "domain" # Fallback
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    filename_base = f"screenshot_{sanitized_domain}_{timestamp_str}.png"
                except Exception as fname_e:
                    print(f"{Fore.YELLOW}Warning: Error generating filename from URL '{current_url}': {fname_e}{Style.RESET_ALL}")
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    filename_base = f"screenshot_{timestamp_str}.png" # Fallback filename

                filename = os.path.join(screenshots_dir, filename_base)

                # Decode base64 data (handle data URI prefix if present)
                img_base64 = ""
                if ',' in img_data_uri:
                    try:
                        # Ensure correct splitting of data URI, e.g. "data:image/png;base64,"
                        header, img_base64 = img_data_uri.split(',', 1)
                        if not header.startswith("data:image/") or ";base64" not in header:
                             print(f"{Fore.YELLOW}Warning: Unexpected data URI header format: {header}{Style.RESET_ALL}")
                    except ValueError:
                         # If split fails, assume it might be raw base64 already
                         print(f"{Fore.YELLOW}Warning: Could not split data URI, assuming raw Base64.{Style.RESET_ALL}")
                         img_base64 = img_data_uri
                else:
                    img_base64 = img_data_uri # Assume raw base64 if no comma

                # Add padding if necessary (robustly check length first)
                if img_base64:
                    missing_padding = len(img_base64) % 4
                    if missing_padding: img_base64 += '=' * (4 - missing_padding)
                else:
                    raise ValueError("Empty Base64 data after processing URI.")


                # Decode and save
                img_bytes = base64.b64decode(img_base64)
                with open(filename, 'wb') as f: f.write(img_bytes)

                relative_path = os.path.relpath(filename, os.getcwd())
                success_msg = f"Screenshot saved to {relative_path}"
                print(f"{Fore.GREEN} {success_msg}{Style.RESET_ALL}")

                # Add file to memory
                memory["files"][relative_path] = {"created": time.time(), "type": "screenshot", "source_url": current_url, "size": len(img_bytes)}
                # self.system_prompt = self._build_system_prompt() # Update prompt (now done at start of loop)

                # Modify result to include the filename in output
                result.output = success_msg
                # Remove the large base64 data from the returned data field
                result.data = {"filename": relative_path, "url": current_url} # Keep relevant info

            except (IndexError, ValueError, base64.binascii.Error, IOError) as e:
                error_msg = f"Error processing or saving screenshot data: {e}"
                print(f"{Fore.RED}[X] {error_msg}{Style.RESET_ALL}")
                # Return error but include original data if helpful? Maybe not if it's huge.
                return ActionResult(status='error', error=error_msg, data={"error_context": str(e)})
            except Exception as e: # Catch other unexpected errors during file save
                 error_msg = f"Unexpected error saving screenshot: {e}"
                 print(f"{Fore.RED}[X] {error_msg}{Style.RESET_ALL}")
                 print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
                 return ActionResult(status='error', error=error_msg, data={"error_context": str(e)})

        elif result.status == 'error':
             # Error already logged by _call_puppeteer_server
             pass # Just return the error result
        elif not (isinstance(result.data, dict) and "data" in result.data):
             # Success status but missing data field
             error_msg = "Puppeteer server indicated success but did not return screenshot data."
             print(f"{Fore.RED}[X] {error_msg}{Style.RESET_ALL}")
             return ActionResult(status='error', error=error_msg, data=result.data)

        return result


    def search_web(self, query: str) -> ActionResult:
        if not query or not isinstance(query, str):
            return ActionResult(status='error', error="Invalid search query provided.")
        query = query.strip()
        if not query:
            return ActionResult(status='error', error="Search query is empty after stripping whitespace.")


        print(f"{Fore.CYAN}[SEARCH] Searching the web for: '{query}'{Style.RESET_ALL}")
        try:
            # URL encode the query properly
            # Use a common search engine - Google might block automated queries frequently. DuckDuckGo might be more lenient.
            # search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
            search_url = f"https://duckduckgo.com/?q={requests.utils.quote(query)}&ia=web" # DDG web search

            # --- Step 1: Browse to search results page ---
            print(f"{Fore.CYAN}Navigating to search engine ({search_url})...{Style.RESET_ALL}")
            browse_result = self.browse_web(search_url)

            if browse_result.status != 'success':
                 # If navigation fails, the whole search fails
                 return ActionResult(status='error', error=f"Search failed: Could not navigate to search engine. ({browse_result.error})")

            # --- Step 2: Wait briefly and take screenshot ---
            # Add a small delay to allow results page to render (may need adjustment)
            wait_time = 4 # seconds (maybe longer for JS-heavy pages)
            print(f"{Fore.CYAN}Waiting {wait_time}s for results to render...{Style.RESET_ALL}")
            time.sleep(wait_time)

            print(f"{Fore.CYAN}Taking screenshot of search results...{Style.RESET_ALL}")
            screenshot_result = self.screenshot_web() # Screenshot the whole page

            # --- Step 3: Process results ---
            if screenshot_result.status == 'success':
                 # Extract saved filename from the successful screenshot result
                 saved_file = "Unknown location" # Default
                 if isinstance(screenshot_result.output, str) and "saved to" in screenshot_result.output:
                      saved_file = screenshot_result.output.split("saved to")[-1].strip()
                 elif isinstance(screenshot_result.data, dict) and "filename" in screenshot_result.data:
                      saved_file = screenshot_result.data["filename"]

                 # Store search results in memory
                 memory["search_results"][query] = {
                     "timestamp": time.time(),
                     "screenshot": saved_file,
                     "url": search_url
                 }

                 output_msg = f"Web search performed for '{query}'. Screenshot of results saved to '{saved_file}'."
                 print(f"{Fore.GREEN} {output_msg}{Style.RESET_ALL}")
                 # Return success, maybe include the filename in data
                 return ActionResult(status='success', output=output_msg, data={"screenshot_file": saved_file})
            else:
                 # Screenshot failed, but browsing worked
                 error_msg = f"Search page loaded, but failed to take screenshot: {screenshot_result.error}"
                 print(f"{Fore.YELLOW}! {error_msg}{Style.RESET_ALL}")
                 # Return an error, but mention that the page was loaded
                 return ActionResult(status='error', error=error_msg, output="Search page was loaded, but screenshot capture failed.")

        except Exception as e:
            error_msg = f"Unexpected error during web search for '{query}': {e}"
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def list_files(self, directory: str = '.', pattern: str = '*') -> ActionResult:
        """Lists files in the specified directory matching the pattern."""
        print(f"{Fore.CYAN}[FILES] Listing files in '{directory}' matching '{pattern}'{Style.RESET_ALL}")
        try:
            # Resolve the directory path safely
            target_dir = self._resolve_path(directory)
            relative_dir = self._get_relative_path(target_dir)

            # Check if directory exists
            if not os.path.exists(target_dir):
                return ActionResult(status='error', error=f"Directory '{relative_dir}' does not exist.")
            if not os.path.isdir(target_dir):
                return ActionResult(status='error', error=f"'{relative_dir}' is not a directory.")

            # Use glob to find matching files
            matching_files = []
            for file_path in glob.glob(os.path.join(target_dir, pattern), recursive=True):
                if os.path.isfile(file_path):
                    rel_path = self._get_relative_path(file_path)
                    file_size = os.path.getsize(file_path)
                    file_time = os.path.getmtime(file_path)
                    matching_files.append({
                        "path": rel_path,
                        "size": file_size,
                        "size_formatted": format_size(file_size),
                        "modified": datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # Update memory with file info if not already known
                    if rel_path not in memory["files"]:
                        memory["files"][rel_path] = {
                            "created": file_time,  # Use mtime as fallback for created time
                            "last_modified": file_time,
                            "size": file_size
                        }

            # Sort files by path
            matching_files.sort(key=lambda x: x["path"])

            # Format output
            if matching_files:
                output_lines = [f"Found {len(matching_files)} file(s) in '{relative_dir}' matching '{pattern}':\n"]
                for file_info in matching_files:
                    output_lines.append(f"{file_info['path']} ({file_info['size_formatted']}, modified: {file_info['modified']})")
                output_text = "\n".join(output_lines)
                print(f"{Fore.GREEN}{output_text}{Style.RESET_ALL}")
                return ActionResult(status='success', output=output_text, data={"files": matching_files})
            else:
                output_text = f"No files found in '{relative_dir}' matching '{pattern}'."
                print(f"{Fore.YELLOW}{output_text}{Style.RESET_ALL}")
                return ActionResult(status='success', output=output_text, data={"files": []})

        except ValueError as ve:
            error_msg = f"Error listing files: {ve}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error listing files: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def find_in_files(self, pattern: str, directory: str = '.', file_pattern: str = '*') -> ActionResult:
        """Searches for a pattern in files matching the file pattern in the specified directory."""
        print(f"{Fore.CYAN}[SEARCH] Finding '{pattern}' in files matching '{file_pattern}' in '{directory}'{Style.RESET_ALL}")
        try:
            # Resolve the directory path safely
            target_dir = self._resolve_path(directory)
            relative_dir = self._get_relative_path(target_dir)

            # Check if directory exists
            if not os.path.exists(target_dir):
                return ActionResult(status='error', error=f"Directory '{relative_dir}' does not exist.")
            if not os.path.isdir(target_dir):
                return ActionResult(status='error', error=f"'{relative_dir}' is not a directory.")

            # Compile the regex pattern
            try:
                regex = re.compile(pattern)
            except re.error as re_err:
                return ActionResult(status='error', error=f"Invalid regex pattern: {re_err}")

            # Find matching files
            matching_files = []
            for file_path in glob.glob(os.path.join(target_dir, file_pattern), recursive=True):
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                            matches = list(regex.finditer(content))
                            if matches:
                                rel_path = self._get_relative_path(file_path)
                                lines = content.splitlines()
                                match_details = []

                                for match in matches:
                                    # Find line number for the match
                                    start_pos = match.start()
                                    line_num = content[:start_pos].count('\n') + 1

                                    # Get the line content
                                    line_content = lines[line_num-1] if line_num <= len(lines) else "<line not found>"

                                    match_details.append({
                                        "line": line_num,
                                        "match": match.group(0)[:100] + ('...' if len(match.group(0)) > 100 else ''),
                                        "content": line_content.strip()[:100] + ('...' if len(line_content) > 100 else '')
                                    })

                                matching_files.append({
                                    "path": rel_path,
                                    "matches": match_details
                                })
                    except UnicodeDecodeError:
                        # Skip binary files
                        pass
                    except Exception as file_err:
                        print(f"{Fore.YELLOW}Warning: Error reading file '{file_path}': {file_err}{Style.RESET_ALL}")

            # Format output
            if matching_files:
                output_lines = [f"Found matches for '{pattern}' in {len(matching_files)} file(s):\n"]
                for file_info in matching_files:
                    output_lines.append(f"File: {file_info['path']} ({len(file_info['matches'])} matches)")
                    for match in file_info['matches'][:5]:  # Limit to first 5 matches per file
                        output_lines.append(f"  Line {match['line']}: {match['content']}")
                    if len(file_info['matches']) > 5:
                        output_lines.append(f"  ... and {len(file_info['matches']) - 5} more matches")
                    output_lines.append("")

                output_text = "\n".join(output_lines)
                print(f"{Fore.GREEN}{output_text}{Style.RESET_ALL}")
                return ActionResult(status='success', output=output_text, data={"files": matching_files})
            else:
                output_text = f"No matches found for '{pattern}' in files matching '{file_pattern}' in '{relative_dir}'."
                print(f"{Fore.YELLOW}{output_text}{Style.RESET_ALL}")
                return ActionResult(status='success', output=output_text, data={"files": []})

        except ValueError as ve:
            error_msg = f"Error searching in files: {ve}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error searching in files: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def execute_python(self, code: str) -> ActionResult:
        """Executes Python code and returns the result."""
        if not CONFIG.get("enable_code_execution", False):
            return ActionResult(status='error', error="Python code execution is disabled in configuration.")

        print(f"{Fore.CYAN}[PYTHON] Executing Python code{Style.RESET_ALL}")
        try:
            # Create a temporary file to execute the code
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False, encoding='utf-8') as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(code)

            # Execute the code and capture output
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=CONFIG.get("command_timeout", 120)
            )

            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_err:
                print(f"{Fore.YELLOW}Warning: Failed to delete temporary file: {cleanup_err}{Style.RESET_ALL}")

            # Store execution result in memory
            execution_id = hashlib.md5(code.encode('utf-8')).hexdigest()[:8]
            memory["code_execution"][execution_id] = {
                "timestamp": time.time(),
                "code": code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

            # Format output
            output_lines = ["Python code execution results:"]
            if result.stdout:
                output_lines.append("\nStandard Output:")
                output_lines.append(result.stdout)
            if result.stderr:
                output_lines.append("\nStandard Error:")
                output_lines.append(result.stderr)
            output_lines.append(f"\nReturn Code: {result.returncode}")

            output_text = "\n".join(output_lines)
            if result.returncode == 0:
                print(f"{Fore.GREEN}{output_text}{Style.RESET_ALL}")
                return ActionResult(status='success', output=output_text, data={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "execution_id": execution_id
                })
            else:
                print(f"{Fore.YELLOW}{output_text}{Style.RESET_ALL}")
                return ActionResult(status='error', error=f"Python code execution failed with return code {result.returncode}",
                                   output=output_text, data={
                                       "stdout": result.stdout,
                                       "stderr": result.stderr,
                                       "returncode": result.returncode,
                                       "execution_id": execution_id
                                   })

        except subprocess.TimeoutExpired:
            error_msg = f"Python code execution timed out after {CONFIG.get('command_timeout', 120)} seconds."
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error executing Python code: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def execute_javascript(self, code: str) -> ActionResult:
        """Executes JavaScript code in the browser context."""
        if not CONFIG.get("enable_code_execution", False):
            return ActionResult(status='error', error="JavaScript code execution is disabled in configuration.")
        if not CONFIG.get("puppeteer_server_enabled", False):
            return ActionResult(status='error', error="Browser server is not enabled. Cannot execute JavaScript.")

        print(f"{Fore.CYAN}[JAVASCRIPT] Executing JavaScript in browser context{Style.RESET_ALL}")
        try:
            # Check if browser is active
            if not memory.get("browser_state"):
                return ActionResult(status='error', error="No active browser session. Use BROWSE_WEB first.")

            # Prepare the request to the puppeteer server
            server_url = CONFIG["puppeteer_server_url"]
            endpoint = f"{server_url}/execute-javascript"

            # Make the request
            response = requests.post(endpoint, json={"code": code})
            response.raise_for_status()
            result = response.json()

            # Store execution result in memory
            execution_id = hashlib.md5(code.encode('utf-8')).hexdigest()[:8]
            memory["code_execution"][execution_id] = {
                "timestamp": time.time(),
                "code": code,
                "result": result
            }

            # Format output
            if result.get("success", False):
                output_text = f"JavaScript execution result:\n{json.dumps(result.get('result'), indent=2)}"
                print(f"{Fore.GREEN}{output_text}{Style.RESET_ALL}")
                return ActionResult(status='success', output=output_text, data={
                    "result": result.get('result'),
                    "execution_id": execution_id
                })
            else:
                error_msg = f"JavaScript execution error: {result.get('error', 'Unknown error')}"
                print(f"{Fore.YELLOW}{error_msg}{Style.RESET_ALL}")
                return ActionResult(status='error', error=error_msg, data={
                    "error": result.get('error'),
                    "execution_id": execution_id
                })

        except requests.RequestException as req_err:
            error_msg = f"Error communicating with browser server: {req_err}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error executing JavaScript: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def copy_to_clipboard(self, text: str) -> ActionResult:
        """Copies text to clipboard."""
        if not CONFIG.get("enable_clipboard", False):
            return ActionResult(status='error', error="Clipboard operations are disabled in configuration.")

        print(f"{Fore.CYAN}[CLIPBOARD] Copying text to clipboard{Style.RESET_ALL}")
        try:
            # Store in memory
            memory["clipboard"] = text

            # Try to use system clipboard if possible
            try:
                import pyperclip
                pyperclip.copy(text)
                system_clipboard = True
            except (ImportError, ModuleNotFoundError):
                # Try to install pyperclip
                try:
                    print(f"{Fore.YELLOW}Installing pyperclip for clipboard support...{Style.RESET_ALL}")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyperclip"])
                    import pyperclip
                    pyperclip.copy(text)
                    system_clipboard = True
                except Exception as install_err:
                    print(f"{Fore.YELLOW}Could not install pyperclip: {install_err}. Using memory-only clipboard.{Style.RESET_ALL}")
                    system_clipboard = False
            except Exception as clip_err:
                print(f"{Fore.YELLOW}Error using system clipboard: {clip_err}. Using memory-only clipboard.{Style.RESET_ALL}")
                system_clipboard = False

            # Format output
            preview = text[:50] + '...' if len(text) > 50 else text
            output_text = f"Text copied to {'system clipboard and ' if system_clipboard else ''}memory clipboard: '{preview}'"
            print(f"{Fore.GREEN}{output_text}{Style.RESET_ALL}")
            return ActionResult(status='success', output=output_text, data={"text": text, "system_clipboard": system_clipboard})

        except Exception as e:
            error_msg = f"Unexpected error copying to clipboard: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def get_clipboard(self) -> ActionResult:
        """Gets the current clipboard content."""
        if not CONFIG.get("enable_clipboard", False):
            return ActionResult(status='error', error="Clipboard operations are disabled in configuration.")

        print(f"{Fore.CYAN}[CLIPBOARD] Getting clipboard content{Style.RESET_ALL}")
        try:
            # Try to get system clipboard if possible
            system_text = None
            try:
                import pyperclip
                system_text = pyperclip.paste()
            except (ImportError, ModuleNotFoundError):
                print(f"{Fore.YELLOW}pyperclip not available. Using memory-only clipboard.{Style.RESET_ALL}")
            except Exception as clip_err:
                print(f"{Fore.YELLOW}Error accessing system clipboard: {clip_err}. Using memory-only clipboard.{Style.RESET_ALL}")

            # Get memory clipboard
            memory_text = memory.get("clipboard")

            # Decide which to use (prefer system clipboard if available and different)
            if system_text is not None and system_text != memory_text:
                text = system_text
                source = "system clipboard"
                # Update memory clipboard to match system
                memory["clipboard"] = system_text
            else:
                text = memory_text
                source = "memory clipboard"

            # Format output
            if text is None:
                output_text = f"Clipboard is empty."
                print(f"{Fore.YELLOW}{output_text}{Style.RESET_ALL}")
                return ActionResult(status='success', output=output_text, data={"text": None, "source": source})
            else:
                preview = text[:50] + '...' if len(text) > 50 else text
                output_text = f"Retrieved text from {source}: '{preview}'"
                print(f"{Fore.GREEN}{output_text}{Style.RESET_ALL}")
                return ActionResult(status='success', output=output_text, data={"text": text, "source": source})

        except Exception as e:
            error_msg = f"Unexpected error getting clipboard content: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def diff_files(self, file1: str, file2: str) -> ActionResult:
        """Shows differences between two files."""
        if not CONFIG.get("enable_file_diff", False):
            return ActionResult(status='error', error="File diff is disabled in configuration.")

        print(f"{Fore.CYAN}[DIFF] Comparing files '{file1}' and '{file2}'{Style.RESET_ALL}")
        try:
            # Resolve the file paths safely
            file1_path = self._resolve_path(file1)
            file2_path = self._resolve_path(file2)
            rel_file1 = self._get_relative_path(file1_path)
            rel_file2 = self._get_relative_path(file2_path)

            # Check if files exist
            if not os.path.exists(file1_path):
                return ActionResult(status='error', error=f"File '{rel_file1}' does not exist.")
            if not os.path.exists(file2_path):
                return ActionResult(status='error', error=f"File '{rel_file2}' does not exist.")
            if not os.path.isfile(file1_path):
                return ActionResult(status='error', error=f"'{rel_file1}' is not a file.")
            if not os.path.isfile(file2_path):
                return ActionResult(status='error', error=f"'{rel_file2}' is not a file.")

            # Read file contents
            try:
                with open(file1_path, 'r', encoding='utf-8', errors='replace') as f1:
                    content1 = f1.readlines()
                with open(file2_path, 'r', encoding='utf-8', errors='replace') as f2:
                    content2 = f2.readlines()
            except UnicodeDecodeError:
                return ActionResult(status='error', error="Cannot diff binary files.")

            # Generate diff
            import difflib
            diff = list(difflib.unified_diff(
                content1, content2,
                fromfile=rel_file1,
                tofile=rel_file2,
                lineterm=''
            ))

            # Format output
            if diff:
                # Colorize diff output
                colored_diff = []
                for line in diff:
                    if line.startswith('+++'):
                        colored_diff.append(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
                    elif line.startswith('---'):
                        colored_diff.append(f"{Fore.RED}{line}{Style.RESET_ALL}")
                    elif line.startswith('+'):
                        colored_diff.append(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
                    elif line.startswith('-'):
                        colored_diff.append(f"{Fore.RED}{line}{Style.RESET_ALL}")
                    elif line.startswith('@@'):
                        colored_diff.append(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
                    else:
                        colored_diff.append(line)

                # Print colored diff
                for line in colored_diff:
                    print(line)

                # Return plain diff for the model
                output_text = "\n".join(diff)
                return ActionResult(status='success', output=output_text, data={"diff": diff})
            else:
                output_text = f"No differences found between '{rel_file1}' and '{rel_file2}'."
                print(f"{Fore.GREEN}{output_text}{Style.RESET_ALL}")
                return ActionResult(status='success', output=output_text, data={"diff": []})

        except Exception as e:
            error_msg = f"Unexpected error comparing files: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def download_file(self, url: str, filename: str) -> ActionResult:
        """Downloads a file from a URL."""
        print(f"{Fore.CYAN}[DOWNLOAD] Downloading file from '{url}' to '{filename}'{Style.RESET_ALL}")
        try:
            # Resolve the file path safely
            target_path = self._resolve_path(filename)
            relative_path = self._get_relative_path(target_path)

            # Check if the directory exists
            target_dir = os.path.dirname(target_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                print(f"{Fore.YELLOW}Created directory: {os.path.dirname(relative_path)}{Style.RESET_ALL}")

            # Download the file with progress bar if enabled
            if CONFIG.get("show_progress_bars", True):
                import requests
                from tqdm import tqdm

                # Make a streaming request
                response = requests.get(url, stream=True)
                response.raise_for_status()

                # Check file size
                file_size = int(response.headers.get('content-length', 0))
                if file_size > CONFIG.get("max_file_size_download", 50 * 1024 * 1024):  # Default 50MB limit
                    return ActionResult(status='error', error=f"File size ({format_size(file_size)}) exceeds the maximum allowed size ({format_size(CONFIG.get('max_file_size_download', 50 * 1024 * 1024))}).")

                # Download with progress bar
                with open(target_path, 'wb') as f, tqdm(
                    desc=f"Downloading {os.path.basename(relative_path)}",
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                            progress_bar.update(len(chunk))
            else:
                # Simple download without progress bar
                import urllib.request
                urllib.request.urlretrieve(url, target_path)

            # Get file info
            file_size = os.path.getsize(target_path)
            file_time = os.path.getmtime(target_path)

            # Store download info in memory
            download_id = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
            memory["downloads"][download_id] = {
                "timestamp": time.time(),
                "url": url,
                "filename": relative_path,
                "size": file_size
            }

            # Update files memory
            memory["files"][relative_path] = {
                "created": file_time,
                "last_modified": file_time,
                "size": file_size
            }

            # Format output
            output_text = f"Downloaded file from '{url}' to '{relative_path}' ({format_size(file_size)})"
            print(f"{Fore.GREEN}{output_text}{Style.RESET_ALL}")
            return ActionResult(status='success', output=output_text, data={
                "url": url,
                "filename": relative_path,
                "size": file_size,
                "size_formatted": format_size(file_size),
                "download_id": download_id
            })

        except Exception as e:
            error_msg = f"Error downloading file: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)

    def summarize_text(self, text: str) -> ActionResult:
        """Summarizes long text."""
        print(f"{Fore.CYAN}[SUMMARIZE] Summarizing text ({len(text)} characters){Style.RESET_ALL}")
        try:
            # Check if text is too short to summarize
            if len(text) < 500:
                return ActionResult(status='success', output="Text is too short to summarize. Original text returned.", data={"summary": text})

            # Use the LLM to summarize the text
            # Create a simplified version of the agent to avoid circular imports
            from google import generativeai as genai

            # Configure the API key
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

            # Create a model instance
            model = genai.GenerativeModel(model_name=CONFIG.get("model", "gemini-2.5-pro-exp-03-25"))

            # Generate the summary
            prompt = f"Please summarize the following text concisely while preserving the key information:\n\n{text}"
            response = model.generate_content(prompt)

            # Extract the summary
            summary = response.text

            # Format output
            output_text = f"Text summarized from {len(text)} to {len(summary)} characters:\n\n{summary}"
            print(f"{Fore.GREEN}Text successfully summarized.{Style.RESET_ALL}")
            return ActionResult(status='success', output=output_text, data={"summary": summary, "original_length": len(text), "summary_length": len(summary)})

        except Exception as e:
            error_msg = f"Error summarizing text: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
            return ActionResult(status='error', error=error_msg)


# --- Main Entry Point ---
if __name__ == "__main__":
    # Initialize colorama for cross-platform colored terminal output
    init()

    # Set up logging if enabled
    if CONFIG.get("enable_logging", False):
        setup_logging()

    # *** Check GEMINI_API_KEY ***
    api_key_env = "GEMINI_API_KEY" # Consistent key name
    if not os.environ.get(api_key_env):
        print(f"{Fore.RED}Fatal Error: {api_key_env} environment variable not set.{Style.RESET_ALL}")
        print(f"Please set it before running.")
        print(f"  Linux/Mac: {Fore.YELLOW}export {api_key_env}='your_api_key'{Style.RESET_ALL}")
        print(f"  Windows Powershell: {Fore.YELLOW}$env:{api_key_env}='your_api_key'{Style.RESET_ALL}")
        print(f"  Windows CMD:        {Fore.YELLOW}set {api_key_env}=your_api_key{Style.RESET_ALL}")
        sys.exit(1)

    agent = None
    exit_code = 0 # Default to success
    try:
        agent = AgentCLI()
        agent.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupt received. Exiting gracefully...{Style.RESET_ALL}")
        if agent: agent.save_memory()
        exit_code = 1 # Indicate interruption
    except EOFError:
         # Handle Ctrl+D or piped input ending
         print(f"\n{Fore.YELLOW}EOF detected. Exiting gracefully...{Style.RESET_ALL}")
         if agent: agent.save_memory()
         exit_code = 0 # EOF is often normal termination
    except SystemExit as se:
         # Allow sys.exit() calls (e.g., from API key check) to pass through
         exit_code = se.code
         if exit_code != 0:
              print(f"{Fore.RED}Exiting due to SystemExit with code {exit_code}{Style.RESET_ALL}")
         # No need to print traceback for SystemExit
    except Exception as e:
        print(f"\n{Fore.RED}{Style.BRIGHT}--- A FATAL UNHANDLED ERROR OCCURRED ---{Style.RESET_ALL}")
        print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
        print(f"{Fore.RED}-----------------------------------------{Style.RESET_ALL}")
        if agent:
            print(f"{Fore.YELLOW}Attempting to save state before final exit...{Style.RESET_ALL}")
            agent.save_memory()
        exit_code = 1 # Indicate error
    finally:
         print(f"\n{Fore.CYAN}AgentCLI session ended.{Style.RESET_ALL}")
         # Exit with the determined code
         sys.exit(exit_code)
