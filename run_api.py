#!/usr/bin/env python3
"""
Startup script for the PDF Question Extractor API
"""

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting PDF Question Extractor API...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔗 API Root: http://localhost:8000")
    print("💊 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "src.api:app",  # Import string instead of app object
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    ) 