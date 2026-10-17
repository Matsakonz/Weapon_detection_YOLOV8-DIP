"""
Real-Time Multi-Camera Weapon Detection & Threat Verification System.
Main entry point launching the Web Dashboard.
"""
import os
import sys
import argparse

# Add project root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.app import start_server

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-Time Weapon Detection & Verification Web Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5001, help="Port to run server on (default: 5001)")
    args = parser.parse_args()

    start_server(host=args.host, port=args.port)
