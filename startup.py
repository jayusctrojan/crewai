#!/usr/bin/env python3
"""
Simplified startup that prioritizes the working CrewAI service
"""

import os
import sys

def main():
    """Start the primary service"""
    print("🚀 Starting CrewAI service...")
    
    # Import and run your existing main.py
    try:
        import main
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
