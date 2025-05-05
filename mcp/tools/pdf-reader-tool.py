"""
PDF Reader Tool for extracting content and metadata from PDF files.

This tool server provides capabilities for reading PDF files, extracting text,
metadata, and performing operations like summarization and text extraction
from specific sections.
"""

import logging
import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional

# Add parent directory to path to import MCP modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mcp.protocol import MCPTool, MCPResource, MCPPrompt
from mcp.server import StandaloneToolServer

try:
    import PyPDF2
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 package not installed. Please install it with: pip install PyPDF2", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

class PDFReaderTool:
    """
    Tool implementations for reading and processing PDF files.
    """
    
    @staticmethod
    async def extract_text(file_path: str, page_numbers: Optional[List[int]] = None) -> Dict:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            page_numbers: Optional list of specific page numbers to extract (1-based)
            
        Returns:
            Dictionary with extracted text
        """
        try:
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            
            # If no specific pages requested, extract all
            if page_numbers is None:
                page_numbers = list(range(1, num_pages + 1))
            
            # Adjust for 0-based indexing
            zero_based_pages = [p - 1 for p in page_numbers if 1 <= p <= num_pages]
            
            # Extract text from specified pages
            pages_text = {}
            for page_idx in zero_based_pages:
                page = reader.pages[page_idx]
                pages_text[f"page_{page_idx + 1}"] = page.extract_text()
            
            return {
                "success": True,
                "num_pages": num_pages,
                "pages_extracted": len(zero_based_pages),
                "text": pages_text
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def get_metadata(file_path: str) -> Dict:
        """
        Extract metadata from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        try:
            reader = PdfReader(file_path)
            
            # Extract metadata
            metadata = reader.metadata
            if metadata:
                # Convert metadata to serializable format
                meta_dict = {
                    key.strip('/'): str(value) 
                    for key, value in metadata.items() 
                    if key and key.startswith('/')
                }
            else:
                meta_dict = {}
            
            # Get basic document info
            num_pages = len(reader.pages)
            
            return {
                "success": True,
                "num_pages": num_pages,
                "metadata": meta_dict
            }
            
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def extract_section(file_path: str, start_page: int, end_page: int) -> Dict:
        """
        Extract text from a section of PDF pages.
        
        Args:
            file_path: Path to the PDF file
            start_page: Starting page number (1-based)
            end_page: Ending page number (1-based)
            
        Returns:
            Dictionary with extracted section text
        """
        try:
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            
            # Validate page ranges
            if start_page < 1 or start_page > num_pages:
                return {
                    "success": False,
                    "error": f"Start page {start_page} out of range (1-{num_pages})"
                }
            
            if end_page < start_page or end_page > num_pages:
                return {
                    "success": False,
                    "error": f"End page {end_page} out of range ({start_page}-{num_pages})"
                }
            
            # Extract text from the section
            section_text = ""
            for page_idx in range(start_page - 1, end_page):
                page = reader.pages[page_idx]
                section_text += page.extract_text() + "\n\n"
            
            return {
                "success": True,
                "start_page": start_page,
                "end_page": end_page,
                "section_text": section_text
            }
            
        except Exception as e:
            logger.error(f"Error extracting section from PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def count_pages(file_path: str) -> Dict:
        """
        Count the number of pages in a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with page count
        """
        try:
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            
            return {
                "success": True,
                "num_pages": num_pages
            }
            
        except Exception as e:
            logger.error(f"Error counting pages in PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

def main():
    """
    Entry point for the PDF Reader Tool server.
    """
    # Define tools
    tools = [
        MCPTool(
            name="pdf_extract_text",
            description="Extract text from a PDF file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF file"
                    },
                    "page_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional list of specific page numbers to extract (1-based)"
                    }
                },
                "required": ["file_path"]
            },
            function=PDFReaderTool.extract_text
        ),
        MCPTool(
            name="pdf_get_metadata",
            description="Extract metadata from a PDF file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF file"
                    }
                },
                "required": ["file_path"]
            },
            function=PDFReaderTool.get_metadata
        ),
        MCPTool(
            name="pdf_extract_section",
            description="Extract text from a section of PDF pages",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF file"
                    },
                    "start_page": {
                        "type": "integer",
                        "description": "Starting page number (1-based)"
                    },
                    "end_page": {
                        "type": "integer",
                        "description": "Ending page number (1-based)"
                    }
                },
                "required": ["file_path", "start_page", "end_page"]
            },
            function=PDFReaderTool.extract_section
        ),
        MCPTool(
            name="pdf_count_pages",
            description="Count the number of pages in a PDF file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF file"
                    }
                },
                "required": ["file_path"]
            },
            function=PDFReaderTool.count_pages
        )
    ]
    
    # Create and run the server
    StandaloneToolServer.create_and_run(
        name="pdf_reader",
        description="Tools for reading and processing PDF files",
        tools=tools
    )

if __name__ == "__main__":
    main()
