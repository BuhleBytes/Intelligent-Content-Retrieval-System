"""
Part 2: Text Processing Pipeline
Assignment: Intelligent Content Retrieval System
Author: Buhle Mlandu

This module processes scraped content from Part 1 into structured chunks
suitable for embedding generation. It handles text cleaning, chunking with
overlap, and metadata preservation.

Requirements:
- Process text into 800-1200 character chunks
- Implement 150+ character overlap between chunks
- Preserve metadata from source documents
- Generate at least 200 total chunks
"""

import json
import glob
import re
import os
import ftfy
from cleantext import clean


def load_all_scraped_data():
    """
    Load all JSON files from Part 1.
    
    Returns:
        list: List of document dictionaries from Part 1
        
    Raises:
        FileNotFoundError: If no JSON files found in data/raw/
        json.JSONDecodeError: If JSON file is corrupted
    """
    all_documents = []
    json_files = glob.glob("data/raw/*.json")
    
    if not json_files:
        raise FileNotFoundError(
            "No JSON files found in data/raw/. "
            "Please run Part 1 first to scrape data."
        )
    
    print(f"Found {len(json_files)} JSON files")

    for filepath in json_files:
        print(f"Loading {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_documents.append(data)
                
            # Display preview (with clean single-line output)
            preview = data['content'][:200].replace('\n', ' ')
            preview = re.sub(r' +', ' ', preview)
            
            print(f"    Category: {data['category']}")
            print(f"    Characters: {data['metadata']['character_count']:,}")
            print(f"    Words: {data['metadata']['word_count']:,}")
            print(f"    Content Prev: {preview}.....")
            print("=" * 70)
            print()
            
        except json.JSONDecodeError as e:
            print(f"❌ Error reading {filepath}: {e}")
            continue
        except KeyError as e:
            print(f"❌ Missing expected field in {filepath}: {e}")
            continue
    
    if not all_documents:
        raise ValueError(
            "No valid documents loaded. "
            "Please check your Part 1 JSON files."
        )
    
    return all_documents


def clean_scraped_content_smart(text):
    """
    Comprehensive cleaning using libraries + custom rules.
    
    Args:
        text (str): Raw text content from web scraping
        
    Returns:
        str: Cleaned text ready for chunking
        
    Notes:
        - Fixes encoding issues (ftfy)
        - Removes URLs, emails, phone numbers
        - Normalizes whitespace and newlines
        - Preserves sentence structure
    """
    if not text:
        return ""
    
    # Step 1: Fix encoding issues
    text = ftfy.fix_text(text)

    # Step 2: Use cleantext for standard cleaning
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_currency_symbols=True,
        no_punct=False,
        no_emoji=True,
        lang="en"
    )
    
    # Step 3: Clean newlines and spacing 
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single newlines → spaces
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single space

    # Step 4: Final cleanup
    text = text.strip()

    return text


def getChunks(text, target_size=1000, overlap=150):
    """
    Split text into overlapping chunks with word boundaries.
    
    Args:
        text (str): Cleaned text to be chunked
        target_size (int): Target chunk size in characters (default: 1000)
        overlap (int): Overlap size in characters (default: 150)
        
    Returns:
        list: List of text chunks (strings)
        
    Features:
        - Breaks at sentence boundaries when possible
        - Falls back to word boundaries (never cuts words)
        - Overlap at word boundaries (doesn't cut words in overlap)
        - No newline characters in chunks
        - Handles edge cases properly
        
    Notes:
        - Minimum chunk size: 100 characters
        - Chunk size range: typically 800-1200 characters
        - Overlap: minimum 150 characters between consecutive chunks
    """
    # Validate input
    if not text or len(text) == 0:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position
        end = min(start + target_size, text_length)
        
        # If not at the very end of text, find a good breaking point
        if end < text_length:
            # STRATEGY 1: Try to break at sentence boundary (. ! ? followed by space)
            # Search in range: [target-200, target+200] for flexibility
            search_start = max(start + 800, end - 200)  # Don't go below 800 chars
            search_end = min(end + 200, text_length)    # Don't exceed text length
            
            # Find the last sentence ending in the search range
            sentence_end = max(
                text.rfind('. ', search_start, search_end),
                text.rfind('! ', search_start, search_end),
                text.rfind('? ', search_start, search_end)
            )
            
            if sentence_end != -1 and sentence_end > start:
                # Found a sentence boundary - use it
                end = sentence_end + 1  # Include the period/punctuation
            else:
                # STRATEGY 2: Fall back to word boundary
                # Find the last space before 'end' (within 100 chars back)
                last_space = text.rfind(' ', max(start + 800, end - 100), end)
                
                if last_space != -1 and last_space > start:
                    end = last_space  # Break at the space
        
        # Extract the chunk
        chunk = text[start:end].strip()
        
        # Only add chunks that have substantial content
        if chunk and len(chunk) >= 100:  # Minimum 100 characters
            chunks.append(chunk)
        
        # Calculate next start position WITH OVERLAP
        if end < text_length:
            # Go back 'overlap' characters from 'end'
            overlap_pos = end - overlap
            
            # Make sure overlap position doesn't cut a word
            # Find the first space AFTER overlap_pos
            if overlap_pos > start:
                next_space = text.find(' ', overlap_pos, end)
                
                if next_space != -1:
                    # Start from the space (beginning of next word)
                    start = next_space + 1
                else:
                    # No space found, just use overlap position
                    start = overlap_pos
            else:
                # Overlap would go before start, just continue from end
                start = end
        else:
            # We've reached the end of the text
            break

        # Safety check: prevent infinite loop
        if start >= end:
            break
    
    return chunks


def writeChunks(all_chunks_data, output_file="data/processed/all_chunks.json"):
    """
    Write all chunks to a JSON file with metadata.
    
    Args:
        all_chunks_data (list): List of dictionaries, each containing:
            - 'chunks': list of chunk strings
            - 'source_doc': original document from Part 1
        output_file (str): Path to output JSON file
    
    Returns:
        dict: Statistics about what was written:
            - output_file: Path to output file
            - total_chunks: Total number of chunks created
            - total_characters: Total character count
            - total_words: Total word count
            - chunks_by_category: Dict of category → chunk count
            
    Raises:
        OSError: If unable to create output directory or write file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create output directory: {e}")
    
    # Build the final chunks list with metadata
    final_chunks = []
    chunk_id_counter = 0
    
    for doc_chunks_data in all_chunks_data:
        chunks_list = doc_chunks_data['chunks']  # List of string chunks
        source_doc = doc_chunks_data['source_doc']  # Original document from Part 1
        
        # Get category identifier
        category = source_doc['category']
        category_id = category.lower().replace(' ', '_')
        
        # Process each chunk
        for i, chunk_text in enumerate(chunks_list):
            # Create chunk dictionary with metadata
            chunk_dict = {
                "chunk_id": f"{category_id}_chunk_{i:03d}",
                "text": chunk_text,
                "metadata": {
                    "source_url": source_doc['url'],
                    "source_category": category,
                    "source_domain": source_doc['domain'],
                    "chunk_index": i,
                    "total_chunks_from_source": len(chunks_list),
                    "character_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "timestamp": source_doc['timestamp']
                }
            }
            
            final_chunks.append(chunk_dict)
            chunk_id_counter += 1
    
    # Write to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_chunks, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise OSError(f"Failed to write output file: {e}")
    
    # Calculate statistics
    stats = {
        'output_file': output_file,
        'total_chunks': len(final_chunks),
        'total_characters': sum(c['metadata']['character_count'] for c in final_chunks),
        'total_words': sum(c['metadata']['word_count'] for c in final_chunks),
        'chunks_by_category': {}
    }
    
    # Count by category
    for chunk in final_chunks:
        category = chunk['metadata']['source_category']
        stats['chunks_by_category'][category] = stats['chunks_by_category'].get(category, 0) + 1
    
    return stats


def main():
    """
    Main execution function for Part 2: Text Processing Pipeline.
    
    Steps:
        1. Load documents from Part 1
        2. Clean and chunk each document
        3. Write chunks to JSON with metadata
        4. Display statistics and validation
        5. Save statistics to separate file
        
    Returns:
        dict: Processing statistics
        
    Raises:
        FileNotFoundError: If Part 1 data not found
        ValueError: If processing fails
    """
    print("=" * 70)
    print("PART 2: TEXT PROCESSING PIPELINE")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Loading documents from Part 1
        print("Step 1: Loading documents from Part 1...")
        documents = load_all_scraped_data()
        print(f"    ✓ Loaded {len(documents)} documents\n")

        # Step 2: Process each document
        print("Step 2: Processing documents into chunks...")
        print()
        all_chunks_data = []

        for doc in documents:
            category = doc['category']
            print(f"Processing: {category}")

            # Clean the content
            cleaned_text = clean_scraped_content_smart(doc['content'])
            print(f"    Cleaned: {len(cleaned_text):,} characters")

            # Get chunks 
            chunks = getChunks(cleaned_text, target_size=1000, overlap=150) 
            print(f"    Chunks created: {len(chunks)}")

            # Store chunks with their source document
            all_chunks_data.append({
                'chunks': chunks,
                'source_doc': doc
            })
            print()

        # Step 3: Writing chunks to JSON file
        print("Step 3: Writing chunks to JSON file...")
        print()

        stats = writeChunks(all_chunks_data, output_file="data/processed/all_chunks.json")
        print(f"    ✓ Saved to: {stats['output_file']}")
        print()
        
        # Step 4: Display statistics
        print("=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)

        print(f"    Total chunks: {stats['total_chunks']}")
        print(f"    Total characters: {stats['total_characters']:,}")
        print(f"    Total words: {stats['total_words']:,}")
        print()

        print("Chunks by category:")
        for category, count in stats['chunks_by_category'].items():
            print(f"  • {category}: {count} chunks")
        print()

        # Step 5: Validation
        print('=' * 70)
        print("VALIDATION")
        print('=' * 70)
        
        if stats['total_chunks'] >= 200:
            print(f"✅ PASSED: {stats['total_chunks']} chunks (requirement: 200+)")
        else:
            print(f"❌ FAILED: Only {stats['total_chunks']} chunks (requirement: 200+)")
            raise ValueError(
                f"Insufficient chunks generated: {stats['total_chunks']} < 200. "
                "Try reducing target_size or increasing overlap."
            )
        
        print()
        
        # Step 6: Save statistics
        stats_file = "data/processed/statistics.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            print(f"✓ Statistics saved to: {stats_file}")
        except OSError as e:
            print(f"⚠️ Warning: Could not save statistics file: {e}")
        
        print()

        print("=" * 70)
        print("✅ PART 2 COMPLETE!")
        print("=" * 70)

        return stats
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please ensure Part 1 has been completed and data exists in data/raw/")
        raise
    
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        raise
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("Please check your data files and try again.")
        raise


if __name__ == "__main__":
    main()