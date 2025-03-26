#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import signal
import argparse

def find_streamlit_processes():
    """Find Streamlit processes and their PIDs."""
    try:
        # Command to find streamlit processes
        cmd = ["ps", "aux", "|", "grep", "streamlit", "|", "grep", "-v", "grep"]
        
        # Use shell=True to allow piping with |
        result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        
        processes = []
        for line in lines:
            if line and "streamlit run" in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    processes.append({
                        'pid': pid,
                        'cmd': line
                    })
        
        return processes
    except Exception as e:
        print(f"Error finding processes: {e}")
        return []

def stop_processes(processes, force=False):
    """Stop the identified streamlit processes."""
    if not processes:
        print("No Streamlit processes found.")
        return
    
    print(f"Found {len(processes)} Streamlit processes:")
    for i, proc in enumerate(processes):
        print(f"{i+1}. PID: {proc['pid']} - {proc['cmd'][:80]}...")
    
    if not force:
        confirm = input("\nDo you want to stop all these processes? [y/N]: ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
    
    for proc in processes:
        try:
            # Try to terminate gracefully first with SIGTERM
            pid = int(proc['pid'])
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to process {pid}")
        except ProcessLookupError:
            print(f"Process {proc['pid']} no longer exists")
        except Exception as e:
            print(f"Error stopping process {proc['pid']}: {e}")

def main():
    """Main function to stop Streamlit app instances."""
    parser = argparse.ArgumentParser(description="Stop running Streamlit processes")
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force kill without confirmation"
    )
    args = parser.parse_args()
    
    processes = find_streamlit_processes()
    stop_processes(processes, args.force)
    
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 