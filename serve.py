#!/usr/bin/env python3
"""Start the rewise server.  Run from the project root: python serve.py"""
import sys
import uvicorn

if __name__ == "__main__":
    reload = "--no-reload" not in sys.argv
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=reload)
