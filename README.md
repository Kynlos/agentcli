# AgentCLI 2.0

<div align="center">

![AgentCLI Logo](https://kynloakari.com/agent/logo.jpeg?text=AgentCLI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Gemini API](https://img.shields.io/badge/API-Gemini-orange)](https://ai.google.dev/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/Kynlos/agentcli.git/agent-cli)

**An advanced autonomous AI assistant for your terminal**

</div>

## 🌟 Overview

AgentCLI is a powerful, autonomous AI assistant that operates in a think-act-observe loop, leveraging Google's Gemini API to accomplish complex tasks directly from your terminal. It's designed for developers, researchers, and power users who want to harness the power of AI in their workflow.

With AgentCLI 2.0, you get enhanced performance, expanded capabilities, and a more intuitive interaction model. The agent can understand complex requests, break them down into manageable steps, and execute them with precision while learning from each interaction.

### Key Features

- **File Operations**: Create, read, edit, and delete files with ease
- **Command Execution**: Run shell commands with full output capture
- **Web Browsing**: Navigate websites and take screenshots
- **Code Execution**: Run Python and JavaScript code directly
- **File Comparison**: Compare files with colorized diff output
- **Clipboard Operations**: Copy and paste text between applications
- **File Downloads**: Download files from the web with progress tracking
- **Text Summarization**: Summarize long text content efficiently
- **Memory Management**: Persistent state across sessions
- **Performance Optimizations**: Response caching, parallel processing, and more

## 📋 Table of Contents

- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [How It Works](#-how-it-works)
- [Technical Architecture](#-technical-architecture)
- [Advanced Features](#-advanced-features)
- [FAQ](#-faq)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 🔧 Installation

### Prerequisites

Before installing AgentCLI, ensure you have the following prerequisites:

- Python 3.8 or higher
- pip (Python package installer)
- A Gemini API key from [Google AI Studio](https://ai.google.dev/)
- Internet connection

### Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/Kynlos/agentcli.git/agent-cli.git
cd agent-cli
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up your API key**

For Linux/macOS:
```bash
export GEMINI_API_KEY='your_api_key_here'
```

For Windows Command Prompt:
```cmd
set GEMINI_API_KEY=your_api_key_here
```

For Windows PowerShell:
```powershell
$env:GEMINI_API_KEY='your_api_key_here'
```

### Optional Components

#### Memory Server

The memory server allows AgentCLI to maintain persistent memory across sessions. To set it up:

```bash
# Install memory server dependencies
pip install fastapi uvicorn

# Start the memory server
python memory_server.py
```

#### Puppeteer Server

The Puppeteer server enables web browsing capabilities. To set it up:

```bash
# Install Node.js and npm if not already installed

# Install Puppeteer server dependencies
cd puppeteer_server
npm install

# Start the Puppeteer server
node server.js
```

## 🚀 Getting Started

### Running AgentCLI

Once you've completed the installation, you can start AgentCLI with a simple command:

```bash
python agent_cli.py
```

### Your First Interaction

After starting AgentCLI, you'll see a welcome message and a prompt for your input. Here's an example of a simple interaction:

```
🤖 AgentCLI 2.0 initialized. How can I help you today?

User: Create a Python script that fetches the current weather for New York

[THINKING] I need to create a Python script that fetches weather data for New York City. I'll use the requests library to make an API call to a weather service.

[ACTION] CREATE_FILE(filename="weather.py", content="import requests\n\ndef get_weather(city):\n    api_key = 'YOUR_API_KEY'  # Replace with your actual API key\n    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'\n    \n    response = requests.get(url)\n    data = response.json()\n    \n    if response.status_code == 200:\n        temp = data['main']['temp']\n        desc = data['weather'][0]['description']\n        print(f'Weather in {city}: {desc}, {temp}°C')\n    else:\n        print(f'Error: {data["message"]}')\n\nif __name__ == '__main__':\n    get_weather('New York')\n")

[OBSERVATION] File created successfully: weather.py

User: Now run the script
```

### Basic Commands

Here are some basic tasks you can ask AgentCLI to perform:

- **File Operations**: "Create a file named hello.txt with 'Hello, World!' as content"
- **Command Execution**: "Run ls -la to show all files in the current directory"
- **Web Browsing**: "Browse to python.org and take a screenshot"
- **Code Execution**: "Run some Python code to check the current date and time"
- **File Search**: "Find all Python files in the current directory"

AgentCLI understands natural language, so you can phrase your requests conversationally.

## ⚙️ Configuration

AgentCLI is highly configurable through the `CONFIG` dictionary in the `agent_cli.py` file. This section explains the key configuration options.

### Configuration Structure

The configuration is organized into logical sections:

```python
CONFIG = {
    # Server connections
    "memory_server_url": "http://localhost:8000",  # URL for the memory server
    "puppeteer_server_url": "http://localhost:3000",  # URL for the web browser server
    "memory_id": "agent_cli_memory_v3",  # Unique identifier for this agent's memory
    "memory_server_enabled": True,  # Set to False if you don't want to use the memory server
    "puppeteer_server_enabled": True,  # Set to False if you don't need web browsing

    # Model configuration
    "model": "gemini-2.5-pro-exp-03-25",  # Gemini model to use
    "model_temperature": 0.4,  # Lower for more deterministic responses, higher for more creative
    "model_max_output_tokens": 8192,  # Maximum tokens in model responses

    # Feature flags - enable or disable specific capabilities
    "enable_code_execution": True,  # Allow Python/JavaScript execution
    "enable_file_diff": True,  # Enable file comparison
    "enable_clipboard": True,  # Enable clipboard operations
    "enable_image_generation": False,  # Disabled by default, requires additional API

    # Performance settings
    "enable_response_cache": True,  # Cache API responses to reduce token usage
    "cache_expiry_seconds": 3600,  # Cache entries expire after 1 hour
    "enable_parallel_processing": True,  # Enable multi-threading for certain operations
    "max_worker_threads": 4,  # Maximum number of worker threads
    "show_progress_bars": True,  # Show progress indicators for long operations

    # Safety settings
    "confirm_risky_actions": True,  # Ask before executing commands or deleting files
    "command_timeout": 120,  # Timeout for command execution in seconds
    "max_file_size_read": 5 * 1024 * 1024,  # 5MB max file size for reading
}
```

### Key Configuration Options

#### Server Connections

- **memory_server_url**: URL for the memory server (default: `http://localhost:8000`)
- **puppeteer_server_url**: URL for the web browser server (default: `http://localhost:3000`)
- **memory_server_enabled**: Set to `False` to disable memory server integration
- **puppeteer_server_enabled**: Set to `False` to disable web browsing capabilities

#### Model Configuration

- **model**: The Gemini model to use (default: `gemini-2.5-pro-exp-03-25`)
- **model_temperature**: Controls randomness in responses (0.0-1.0)
- **model_max_output_tokens**: Maximum tokens in model responses

#### Feature Flags

- **enable_code_execution**: Allow Python/JavaScript code execution
- **enable_file_diff**: Enable file comparison functionality
- **enable_clipboard**: Enable clipboard operations
- **enable_image_generation**: Enable image generation (requires additional API)

#### Performance Settings

- **enable_response_cache**: Cache API responses to reduce token usage
- **enable_parallel_processing**: Enable multi-threading for certain operations
- **show_progress_bars**: Show progress indicators for long operations

#### Safety Settings

- **confirm_risky_actions**: Ask before executing commands or deleting files
- **command_timeout**: Timeout for command execution in seconds
- **max_file_size_read**: Maximum file size for reading operations

## 📝 Usage Examples

This section provides detailed examples of how to use AgentCLI for various tasks.

### File Operations

#### Creating and Editing Files

```
User: Create a Python script called calculator.py that can add, subtract, multiply, and divide two numbers

[THINKING] I'll create a simple calculator script in Python with functions for basic arithmetic operations.

[ACTION] CREATE_FILE(filename="calculator.py", content="def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n\ndef multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    if b == 0:\n        return 'Error: Division by zero'\n    return a / b\n\ndef calculator():\n    print('Simple Calculator')\n    print('1. Add')\n    print('2. Subtract')\n    print('3. Multiply')\n    print('4. Divide')\n    \n    choice = input('Enter choice (1-4): ')\n    num1 = float(input('Enter first number: '))\n    num2 = float(input('Enter second number: '))\n    \n    if choice == '1':\n        print(f'{num1} + {num2} = {add(num1, num2)}')\n    elif choice == '2':\n        print(f'{num1} - {num2} = {subtract(num1, num2)}')\n    elif choice == '3':\n        print(f'{num1} * {num2} = {multiply(num1, num2)}')\n    elif choice == '4':\n        print(f'{num1} / {num2} = {divide(num1, num2)}')\n    else:\n        print('Invalid input')\n\nif __name__ == '__main__':\n    calculator()")

[OBSERVATION] File created successfully: calculator.py

User: Add a function to calculate the power of a number

[THINKING] I need to add a power function to the calculator.py file. I'll first read the current content, then add the new function and update the calculator menu.

[ACTION] READ_FILE(filename="calculator.py")

[OBSERVATION] [File content displayed]

[ACTION] EDIT_FILE(filename="calculator.py", new_content="def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n\ndef multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    if b == 0:\n        return 'Error: Division by zero'\n    return a / b\n\ndef power(a, b):\n    return a ** b\n\ndef calculator():\n    print('Simple Calculator')\n    print('1. Add')\n    print('2. Subtract')\n    print('3. Multiply')\n    print('4. Divide')\n    print('5. Power')\n    \n    choice = input('Enter choice (1-5): ')\n    num1 = float(input('Enter first number: '))\n    num2 = float(input('Enter second number: '))\n    \n    if choice == '1':\n        print(f'{num1} + {num2} = {add(num1, num2)}')\n    elif choice == '2':\n        print(f'{num1} - {num2} = {subtract(num1, num2)}')\n    elif choice == '3':\n        print(f'{num1} * {num2} = {multiply(num1, num2)}')\n    elif choice == '4':\n        print(f'{num1} / {num2} = {divide(num1, num2)}')\n    elif choice == '5':\n        print(f'{num1} ^ {num2} = {power(num1, num2)}')\n    else:\n        print('Invalid input')\n\nif __name__ == '__main__':\n    calculator()")

[OBSERVATION] File edited successfully: calculator.py
```

#### Searching for Files and Content

```
User: Find all Python files in the current directory

[THINKING] I'll use the LIST_FILES action to find all Python files in the current directory.

[ACTION] LIST_FILES(pattern="*.py")

[OBSERVATION] Found 3 file(s) in '.' matching '*.py':
calculator.py (1.2 KB, modified: 2023-06-15 14:32:45)
agent_cli.py (125.4 KB, modified: 2023-06-15 10:15:22)
weather.py (512 bytes, modified: 2023-06-15 14:30:11)

User: Find all functions in calculator.py

[THINKING] I'll use the FIND_IN_FILES action to search for function definitions in calculator.py.

[ACTION] FIND_IN_FILES(pattern="def ", file_pattern="calculator.py")

[OBSERVATION] Found matches for 'def ' in 1 file(s):
File: calculator.py (6 matches)
  Line 1: def add(a, b):
  Line 4: def subtract(a, b):
  Line 7: def multiply(a, b):
  Line 10: def divide(a, b):
  Line 14: def power(a, b):
  Line 17: def calculator():
```

### Web Browsing and Interaction

```
User: Browse to Python.org and take a screenshot

[THINKING] I'll navigate to the Python.org website and take a screenshot of the page.

[ACTION] BROWSE_WEB(url="https://www.python.org")

[OBSERVATION] Successfully navigated to https://www.python.org

[ACTION] SCREENSHOT_WEB()

[OBSERVATION] Screenshot saved to 'screenshot_20230615_143545.png'

User: Extract the latest Python version from the website

[THINKING] I'll use JavaScript execution to extract the latest Python version from the website.

[ACTION] EXECUTE_JAVASCRIPT(code="document.querySelector('.download-for-current-os .release-number').textContent")

[OBSERVATION] JavaScript execution result:
"Python 3.11.4"
```

### Command Execution

```
User: Check the current directory structure

[THINKING] I'll use the EXECUTE_COMMAND action to run the 'ls -la' command (or 'dir' on Windows) to show the directory structure.

[ACTION] EXECUTE_COMMAND(command="ls -la")

[OBSERVATION] total 156
drwxr-xr-x  5 user user  4096 Jun 15 14:45 .
drwxr-xr-x 18 user user  4096 Jun 15 10:00 ..
-rw-r--r--  1 user user  1213 Jun 15 14:32 calculator.py
-rw-r--r--  1 user user   512 Jun 15 14:30 weather.py
-rw-r--r--  1 user user 125432 Jun 15 10:15 agent_cli.py
-rw-r--r--  1 user user  15678 Jun 15 14:35 screenshot_20230615_143545.png

User: Create a new directory called 'projects' and move the Python files there

[THINKING] I'll create a new directory called 'projects' and move the Python files there using shell commands.

[ACTION] EXECUTE_COMMAND(command="mkdir -p projects")

[OBSERVATION] Command executed successfully.

[ACTION] EXECUTE_COMMAND(command="mv *.py projects/")

[OBSERVATION] Command executed successfully.

[ACTION] EXECUTE_COMMAND(command="ls -la projects/")

[OBSERVATION] total 12
drwxr-xr-x 2 user user 4096 Jun 15 14:47 .
drwxr-xr-x 6 user user 4096 Jun 15 14:47 ..
-rw-r--r-- 1 user user 1213 Jun 15 14:32 calculator.py
-rw-r--r-- 1 user user  512 Jun 15 14:30 weather.py
-rw-r--r-- 1 user user 125432 Jun 15 10:15 agent_cli.py
```

### Code Execution

```
User: Run some Python code to check system information

[THINKING] I'll use the EXECUTE_PYTHON action to run Python code that retrieves system information.

[ACTION] EXECUTE_PYTHON(code="import platform\nimport sys\n\nprint(f'Python version: {sys.version}')\nprint(f'Platform: {platform.platform()}')\nprint(f'Processor: {platform.processor()}')")

[OBSERVATION] Python code execution results:
Standard Output:
Python version: 3.9.7 (default, Sep 16 2021, 13:09:58)
[GCC 7.5.0]
Platform: Linux-5.4.0-1065-aws-x86_64-with-glibc2.31
Processor: x86_64
Return Code: 0
```

### Advanced Features

#### File Comparison

```
User: Create two versions of a config file and show the differences

[THINKING] I'll create two versions of a configuration file and then use the DIFF_FILES action to show the differences.

[ACTION] CREATE_FILE(filename="config_v1.json", content="{\n  \"version\": \"1.0\",\n  \"settings\": {\n    \"debug\": false,\n    \"timeout\": 30,\n    \"max_retries\": 3\n  },\n  \"features\": [\n    \"basic\",\n    \"advanced\"\n  ]\n}")

[OBSERVATION] File created successfully: config_v1.json

[ACTION] CREATE_FILE(filename="config_v2.json", content="{\n  \"version\": \"2.0\",\n  \"settings\": {\n    \"debug\": true,\n    \"timeout\": 60,\n    \"max_retries\": 5,\n    \"new_setting\": \"value\"\n  },\n  \"features\": [\n    \"basic\",\n    \"advanced\",\n    \"experimental\"\n  ]\n}")

[OBSERVATION] File created successfully: config_v2.json

[ACTION] DIFF_FILES(file1="config_v1.json", file2="config_v2.json")

[OBSERVATION] --- config_v1.json
+++ config_v2.json
@@ -1,9 +1,10 @@
 {
-  "version": "1.0",
+  "version": "2.0",
   "settings": {
-    "debug": false,
-    "timeout": 30,
-    "max_retries": 3
+    "debug": true,
+    "timeout": 60,
+    "max_retries": 5,
+    "new_setting": "value"
   },
   "features": [
     "basic",
-    "advanced"
+    "advanced",
+    "experimental"
   ]
 }
```

#### Text Summarization

```
User: Summarize a long article

[THINKING] I'll demonstrate the text summarization capability by summarizing a long text.

[ACTION] SUMMARIZE_TEXT(text="The field of artificial intelligence has seen remarkable progress in recent years, particularly in the area of large language models. These models, trained on vast amounts of text data, have demonstrated impressive capabilities in understanding and generating human language. They can write essays, answer questions, translate languages, and even generate code. However, these advancements come with significant challenges. The computational resources required to train these models are substantial, raising concerns about environmental impact and accessibility. Additionally, these models can sometimes generate incorrect or misleading information, a phenomenon often referred to as 'hallucination.' They may also reflect and potentially amplify biases present in their training data. As these technologies continue to evolve, it is crucial to address these challenges through ongoing research and responsible deployment practices. The future of AI will likely involve finding the right balance between leveraging the capabilities of these powerful models while mitigating their limitations and potential risks.")

[OBSERVATION] Text summarized from 1023 to 215 characters:

Recent advances in AI language models show impressive capabilities in understanding and generating human language, but face challenges including resource requirements, potential for generating incorrect information, and bias concerns. Ongoing research and responsible deployment are needed to balance capabilities with limitations and risks.
```

## 🔍 How It Works

AgentCLI operates in a sophisticated think-act-observe loop, leveraging Google's Gemini API to accomplish tasks autonomously with remarkable precision and adaptability.

### Think-Act-Observe Loop

#### 1. Think

In the thinking phase, AgentCLI:

- Analyzes the user's request and current context
- Considers available tools and previous observations
- Formulates a plan of action with specific steps
- Anticipates potential challenges and prepares fallback strategies

The agent leverages Gemini's advanced reasoning capabilities to understand context, identify required steps, and develop an optimal execution strategy.

#### 2. Act

In the action phase, AgentCLI:

- Executes a specific action from its repertoire of capabilities
- Provides precise parameters derived from the thinking phase
- Handles complex operations with appropriate error handling
- Manages resources efficiently for long-running operations

The agent can perform 17+ different actions, from simple file operations to complex web interactions and code execution.

#### 3. Observe

In the observation phase, AgentCLI:

- Captures the results of the executed action
- Analyzes success or failure conditions
- Updates its understanding of the environment
- Stores observations in memory for future reference

This critical feedback loop allows the agent to adapt to unexpected outcomes, recover from errors, and refine its approach for subsequent actions.

### Memory System

AgentCLI maintains a comprehensive memory system that tracks:

- Conversation history with the user
- File operations and their results
- Web browsing state and screenshots
- Command execution history and outputs
- Code execution results and outputs

This memory system allows the agent to build context across multiple interactions, refer back to previous actions, and make informed decisions based on the evolving state of the environment.

### Workflow Example

```
User Input → Agent Thinking → Action Execution → Observation → Agent Thinking → ...
```

For example, when asked to "Create a Python script that fetches weather data":

1. **Think**: The agent analyzes the request, determines it needs to create a file with Python code that uses a weather API.
2. **Act**: The agent executes `CREATE_FILE` with appropriate filename and content parameters.
3. **Observe**: The agent captures the result (file created successfully) and updates its memory.
4. **Think**: The agent considers what to do next, perhaps suggesting to run the script or modify it.

This loop continues until the task is completed or the agent encounters an insurmountable obstacle.

## 🛠 Technical Architecture

AgentCLI is built with a modular, extensible architecture designed for performance, reliability, and ease of use.

### Core Components

#### LLM Integration

AgentCLI integrates with Google's Gemini API through a robust client that handles:

- API authentication and request formatting
- Response parsing and error handling
- Streaming responses for improved user experience
- Caching to reduce API calls and improve performance

The integration is designed to be resilient to API changes and to handle rate limiting gracefully.

#### Memory System

The memory system consists of several components:

- **Conversation Memory**: Stores the history of user inputs and agent responses
- **File Memory**: Tracks file operations and maintains metadata about files
- **Browser Memory**: Stores the current state of web browsing sessions
- **Execution Memory**: Records command and code execution history and results

Memory can be persisted across sessions using the optional memory server, allowing for long-running interactions.

#### Action Framework

The action framework provides a unified interface for all agent capabilities:

- **Action Parser**: Parses action strings into structured commands
- **Action Executor**: Routes actions to appropriate handlers
- **Action Handlers**: Implement specific functionality for each action type
- **Result Formatter**: Formats action results for observation

This modular design makes it easy to add new actions or modify existing ones without changing the core loop.

#### Web Integration

Web integration is provided through a Puppeteer-based server that enables:

- Headless web browsing with full JavaScript support
- Screenshot capture of entire pages or specific elements
- JavaScript execution in the browser context
- Web search capabilities

### Advanced Features

#### Response Caching

AgentCLI implements an intelligent caching system that:

- Stores API responses keyed by conversation context
- Applies configurable expiration policies
- Reduces token usage and improves response times
- Handles cache invalidation when context changes significantly

#### Parallel Processing

Long-running operations benefit from parallel processing capabilities:

- Multi-threaded execution for I/O-bound operations
- Progress tracking with visual feedback
- Cancellation support for interruptible operations
- Resource management to prevent thread exhaustion

#### Safety Mechanisms

Several safety features are built into AgentCLI:

- Confirmation prompts for potentially risky actions
- File size limits for reading and writing operations
- Execution timeouts to prevent hanging operations
- Path resolution safety to prevent directory traversal
- Input validation to prevent injection attacks

### System Requirements

#### Minimum Requirements

- Python 3.8+
- 4GB RAM
- 1GB free disk space
- Internet connection

#### Recommended Requirements

- Python 3.9+
- 8GB RAM
- 2GB free disk space
- Broadband internet connection

#### Dependencies

- Google Generative AI Python SDK
- Colorama (for terminal colors)
- Requests (for HTTP operations)
- Optional: tqdm (for progress bars)
- Optional: Pillow (for image processing)

## ✨ Advanced Features

AgentCLI includes several advanced features that extend its capabilities beyond basic operations.

### File Diff Visualization

The file diff feature allows you to compare two files and visualize the differences between them. This is particularly useful for:

- Comparing different versions of configuration files
- Reviewing code changes
- Identifying differences between data exports

The diff output is colorized and formatted for easy reading, with additions, deletions, and changes clearly marked.

### Code Execution Sandbox

AgentCLI can execute Python and JavaScript code in a controlled environment:

- **Python Execution**: Run Python code with full access to standard libraries
- **JavaScript Execution**: Execute JavaScript in the browser context when web browsing is enabled

Code execution is performed in a sandboxed environment with configurable timeouts and resource limits to prevent runaway processes.

### Clipboard Integration

The clipboard integration feature allows AgentCLI to interact with the system clipboard:

- **Copy to Clipboard**: Copy text to the system clipboard for use in other applications
- **Get from Clipboard**: Retrieve text from the system clipboard for use in AgentCLI

This feature is particularly useful for transferring data between AgentCLI and other applications.

### Text Summarization

The text summarization feature uses the Gemini API to condense long text into concise summaries while preserving key information. This is useful for:

- Summarizing articles or documentation
- Extracting key points from long discussions
- Creating executive summaries of reports

### Progress Tracking

For long-running operations, AgentCLI provides visual progress tracking:

- Progress bars for file downloads
- Status indicators for web operations
- Elapsed time tracking for command execution

This provides feedback to the user and makes it easier to monitor the progress of complex tasks.

### Parallel Processing

AgentCLI can perform certain operations in parallel to improve performance:

- Multi-threaded file operations
- Concurrent web requests
- Background processing for non-blocking operations

Parallel processing is managed through a thread pool with configurable limits to prevent resource exhaustion.

## ❓ FAQ

### What is AgentCLI?

AgentCLI is an advanced autonomous AI assistant that operates in a think-act-observe loop. It can perform various tasks such as file operations, command execution, web browsing, code execution, and more, all from your terminal. It uses the Gemini API to power its AI capabilities, providing state-of-the-art reasoning and problem-solving abilities.

Unlike simple command-line tools, AgentCLI can understand complex natural language requests, break them down into manageable steps, and execute them autonomously while adapting to changing conditions and learning from its observations. It maintains a sophisticated memory system that allows it to build context across multiple interactions.

### Is AgentCLI secure?

AgentCLI has built-in safety features such as confirmation prompts for risky actions like executing commands or deleting files. It also implements file size limits, execution timeouts, and careful path resolution to prevent unintended consequences.

However, as with any tool that can execute commands, you should use it responsibly and be aware of the actions it's taking. You can adjust the safety settings in the configuration to match your comfort level, including disabling potentially risky features entirely or requiring confirmation for more action types.

The agent is designed to be transparent about its actions, showing you what it's thinking and doing at each step, so you can always intervene if needed.

### Do I need an API key?

Yes, you need a Gemini API key to use AgentCLI. You can obtain one from the [Google AI Studio](https://ai.google.dev/) website. Once you have your API key, you can set it as an environment variable as shown in the installation section.

The API key is used to authenticate with the Gemini API and is required for all operations that involve the AI model.

### Can I extend AgentCLI with custom actions?

Yes, AgentCLI is designed to be extensible. You can add custom actions by:

1. Implementing a new method in the `AgentCLI` class
2. Adding the action to the system prompt
3. Adding parsing logic to the `execute_action` method

The modular design makes it easy to add new capabilities without modifying the core loop. See the [Contributing](#-contributing) section for more information on how to contribute to the project.

### What operating systems are supported?

AgentCLI is designed to work on Windows, macOS, and Linux. It includes platform-specific adaptations for features like clipboard operations and terminal handling to ensure a consistent experience across different operating systems.

The agent automatically detects your operating system and adjusts its behavior accordingly, such as using the appropriate command syntax and file path conventions. This cross-platform compatibility makes it a versatile tool for developers working in mixed environments.

### How is performance optimized in AgentCLI?

AgentCLI 2.0 includes several performance optimizations to ensure smooth operation even with complex tasks:

- Response caching reduces API calls by storing and reusing responses for similar queries
- Parallel processing enables multi-threaded execution for long-running operations
- Progress tracking with tqdm provides visual feedback for time-consuming tasks
- Streaming responses from the API allows for faster initial response times
- Memory pruning prevents excessive memory usage during long sessions
- Configurable timeouts prevent operations from hanging indefinitely

These optimizations make AgentCLI responsive and efficient, even when handling resource-intensive tasks like file searches across large codebases or complex web interactions.

## 💪 Contributing

Contributions to AgentCLI are welcome! Here's how you can contribute:

### Reporting Issues

If you encounter a bug or have a feature request, please open an issue on the GitHub repository. When reporting issues, please include:

- A clear and descriptive title
- A detailed description of the issue or feature request
- Steps to reproduce the issue (for bugs)
- Expected behavior and actual behavior
- Any relevant logs or screenshots

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/Kynlos/agentcli.git/agent-cli.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -r requirements-dev.txt`

### Making Changes

1. Make your changes to the codebase
2. Add or update tests as necessary
3. Run the tests: `pytest`
4. Ensure your code follows the project's coding style
5. Commit your changes with a descriptive commit message

### Submitting a Pull Request

1. Push your changes to your fork: `git push origin feature/your-feature-name`
2. Open a pull request against the main repository
3. Provide a clear description of the changes and any relevant issue numbers
4. Wait for the maintainers to review your pull request

### Adding New Actions

To add a new action to AgentCLI:

1. Implement the action method in the `AgentCLI` class
2. Add the action to the system prompt in the `_build_system_prompt` method
3. Add parsing logic to the `execute_action` method
4. Add tests for the new action
5. Update the documentation to include the new action

### Code Style

Please follow these guidelines when contributing code:

- Use descriptive variable names and comments
- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and methods
- Keep functions focused on a single responsibility
- Add type hints where appropriate

## 🔐 License

AgentCLI is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Kynlo Akari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👏 Acknowledgements

- [Google Gemini API](https://ai.google.dev/) for providing the AI capabilities
- [Colorama](https://pypi.org/project/colorama/) for terminal colors
- [Requests](https://requests.readthedocs.io/) for HTTP operations
- [tqdm](https://github.com/tqdm/tqdm) for progress bars
- [Puppeteer](https://pptr.dev/) for web browsing capabilities
- All contributors who have helped improve AgentCLI

---

<div align="center">

Made with ❤️ by Kynlo Akari

</div>
