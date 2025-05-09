import asyncio
from mcp_clients.linear_client import LinearMCPClient

async def main():
    client = LinearMCPClient()
    await client.connect_to_server()

import os
import subprocess
import time
import select
import json
import glob

def test_mcp_remote_simple():
    # Clear any existing auth data
    os.system("rm -rf ~/.mcp-auth")
    
    # This is the exact command format from Claude Desktop config
    cmd = [
        "npx",  # Use system npx
        "-y", 
        "mcp-remote", 
        "https://mcp.linear.app/sse"
    ]
    
    # Environment variables similar to Claude Desktop
    env = {
        "LINEAR_TOKEN": os.getenv("LINEAER_API_KEY"),
        "PATH": os.environ.get("PATH", "")
    }
    
    # Start the process and just let it run
    print("Starting mcp-remote process")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Set up polling
    poll = select.poll()
    poll.register(process.stdout, select.POLLIN)
    poll.register(process.stderr, select.POLLIN)
    
    # Monitor output with timeout
    try:
        connection_time = time.time()
        timeout = 30  # 30 second timeout
        
        while True:
            # Check if there's data to read (with a 100ms timeout)
            ready = poll.poll(100)
            
            for fd, event in ready:
                if fd == process.stdout.fileno():
                    line = process.stdout.readline()
                    if line:
                        print(f"STDOUT: {line.strip()}")
                elif fd == process.stderr.fileno():
                    line = process.stderr.readline()
                    if line:
                        print(f"STDERR: {line.strip()}")
            
            # Check if process has exited
            if process.poll() is not None:
                print(f"Process exited with code {process.returncode}")
                break
            
            # Print a waiting message every second
            if int(time.time()) % 1 == 0:
                print(f"Waiting for process... {int(time.time() - connection_time)}s elapsed")
            
            # Check timeout
            if time.time() - connection_time > timeout:
                print(f"Process timed out after {timeout} seconds")
                process.kill()
                break
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Test interrupted, killing process")
        process.kill()
    finally:
        # Clean up
        if process.poll() is None:
            process.kill()

def test_mcp_remote_with_extracted_token():
    """Test connecting to Linear using the OAuth token explicitly extracted from storage"""
    
    # Find and load the token file
    print("Checking for existing OAuth tokens")
    auth_dir = os.path.expanduser("~/.mcp-auth")
    token_files = glob.glob(f"{auth_dir}/**/*_tokens.json", recursive=True)
    
    if not token_files:
        print("No existing OAuth tokens found in ~/.mcp-auth")
        print("Please run test_mcp_remote_simple() first to complete the OAuth flow")
        return
    
    # Extract the access token
    access_token = None
    for token_file in token_files:
        try:
            with open(token_file, 'r') as f:
                token_data = json.load(f)
                if 'access_token' in token_data:
                    access_token = token_data['access_token']
                    print(f"Using access token: {access_token[:5]}...{access_token[-5:]}")
                    break
        except Exception as e:
            print(f"Error reading token file {token_file}: {e}")
    
    if not access_token:
        print("No access token found in token files")
        return
    
    # Use the token directly as a header
    cmd = [
        "npx",
        "-y", 
        "mcp-remote", 
        "https://mcp.linear.app/sse",
        "--header", 
        f"Authorization:Bearer {access_token}",  # Pass token directly as header
        "--transport", 
        "sse-only",  # Force SSE transport
        "--verbose",
        "--debug"
    ]
    
    # Environment variables - don't include the LINEAR_TOKEN
    env = {
        "PATH": os.environ.get("PATH", ""),
        # Intentionally NOT using MCP_REMOTE_CONFIG_DIR to avoid token conflicts
        "MCP_REMOTE_CONFIG_DIR": "/tmp/mcp-linear-test"  # Use a fresh directory
    }
    
    # Start the process
    print("\nStarting mcp-remote process with extracted OAuth token")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Set up polling
    poll = select.poll()
    poll.register(process.stdout, select.POLLIN)
    poll.register(process.stderr, select.POLLIN)
    
    # Monitor output with timeout
    try:
        connection_time = time.time()
        timeout = 30  # 30 second timeout
        
        while True:
            # Check if there's data to read (with a 100ms timeout)
            ready = poll.poll(100)
            
            for fd, event in ready:
                if fd == process.stdout.fileno():
                    line = process.stdout.readline()
                    if line:
                        print(f"STDOUT: {line.strip()}")
                elif fd == process.stderr.fileno():
                    line = process.stderr.readline()
                    if line:
                        print(f"STDERR: {line.strip()}")
            
            # Check if process has exited
            if process.poll() is not None:
                print(f"Process exited with code {process.returncode}")
                break
            
            # Print a waiting message every second
            current_time = time.time()
            if int(current_time) % 1 == 0:
                elapsed = int(current_time - connection_time)
                print(f"Waiting for process... {elapsed}s elapsed")
            
            # Check timeout
            if time.time() - connection_time > timeout:
                print(f"Process timed out after {timeout} seconds")
                process.kill()
                break
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Test interrupted, killing process")
        process.kill()
    finally:
        # Clean up
        if process.poll() is None:
            process.kill()

def test_mcp_remote_with_token_and_request():
    """Test connecting with the extracted token and send a tools/list request to Linear"""
    
    # Find and load the token file
    print("Checking for existing OAuth tokens")
    auth_dir = os.path.expanduser("~/.mcp-auth")
    token_files = glob.glob(f"{auth_dir}/**/*_tokens.json", recursive=True)
    
    if not token_files:
        print("No existing OAuth tokens found in ~/.mcp-auth")
        return
    
    # Extract the access token
    access_token = None
    for token_file in token_files:
        try:
            with open(token_file, 'r') as f:
                token_data = json.load(f)
                if 'access_token' in token_data:
                    access_token = token_data['access_token']
                    print(f"Using access token: {access_token[:5]}...{access_token[-5:]}")
                    break
        except Exception as e:
            print(f"Error reading token file {token_file}: {e}")
    
    if not access_token:
        print("No access token found in token files")
        return
    
    # Use the token directly as a header
    cmd = [
        "npx",
        "-y", 
        "mcp-remote", 
        "https://mcp.linear.app/sse",
        "--header", 
        f"Authorization:Bearer {access_token}",
        "--transport", 
        "sse-only"
    ]
    
    # Environment variables
    env = {
        "PATH": os.environ.get("PATH", ""),
        # Intentionally NOT using MCP_REMOTE_CONFIG_DIR to avoid token conflicts
        "MCP_REMOTE_CONFIG_DIR": "/tmp/mcp-linear-test"  # Use a fresh directory
    }
    
    # Start the process
    print("\nStarting mcp-remote process with extracted OAuth token")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Set up polling
    poll = select.poll()
    poll.register(process.stdout, select.POLLIN)
    poll.register(process.stderr, select.POLLIN)
    
    # Wait for connection to be established
    print("Waiting for connection to be established...")
    connection_established = False
    start_time = time.time()
    
    while not connection_established and time.time() - start_time < 15:
        ready = poll.poll(100)
        for fd, event in ready:
            if fd == process.stderr.fileno():
                line = process.stderr.readline()
                if line:
                    print(f"STDERR: {line.strip()}")
                    if "Connected to remote server" in line or "Proxy established successfully" in line:
                        connection_established = True
                        print("✅ Connection established!")
                        break
            elif fd == process.stdout.fileno():
                line = process.stdout.readline()
                if line:
                    print(f"STDOUT: {line.strip()}")
        time.sleep(0.1)
    
    if not connection_established:
        print("Failed to establish connection within timeout")
        process.kill()
        return
    
    # Wait a moment for the proxy to be fully established
    time.sleep(1)
    
    # Send a request to list tools
    print("\nSending request to list tools")
    tools_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    # Send the request
    process.stdin.write(json.dumps(tools_request) + "\n")
    process.stdin.flush()
    
    # Wait for the response
    print("Waiting for tools response...")
    start_time = time.time()
    
    while time.time() - start_time < 15:  # Extended timeout
        ready = poll.poll(100)
        for fd, event in ready:
            if fd == process.stdout.fileno():
                line = process.stdout.readline()
                if line:
                    print(f"RESPONSE: {line.strip()}")
                    try:
                        response = json.loads(line)
                        if "result" in response and "tools" in response["result"]:
                            print("\n✅ Linear MCP tools available:")
                            for tool in response["result"]["tools"]:
                                print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
                            
                            # Success! We've verified the connection works
                            print("\n🎉 Success! OAuth token connection to Linear MCP is working!")
                            
                            # Cleanly shut down
                            shutdown_request = {
                                "jsonrpc": "2.0",
                                "id": 2,
                                "method": "shutdown",
                                "params": {}
                            }
                            
                            process.stdin.write(json.dumps(shutdown_request) + "\n")
                            process.stdin.flush()
                            
                            # Allow time for shutdown to complete
                            time.sleep(1)
                            return
                    except json.JSONDecodeError:
                        print("Response was not valid JSON")
            elif fd == process.stderr.fileno():
                line = process.stderr.readline()
                if line:
                    print(f"STDERR: {line.strip()}")
        time.sleep(0.1)
    
    print("Did not receive tools response within timeout")
    
    # Clean up
    try:
        process.kill()
    except:
        pass

if __name__ == "__main__":
    test_mcp_remote_with_token_and_request()