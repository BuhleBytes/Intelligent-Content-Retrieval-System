import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

def load_chunks(filepath="data/processed/all_chunks.json"):
    """ 
    Load processed chunks from Part 2

    Returns: A list of chunk dictionaries
    """
    print(f"Loading chunks from {filepath}....")
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    print(f"Loaded {len(chunks_data)} chunks")

    #Display a sample
    if chunks_data:
        sample = chunks_data[0]
        print(f"    Sample chuck id: {sample['chunk_id']}")
        print(f"    Sample category: {sample['metadata']['source_category']}")
        print(f"    Sample text preview: {sample['text'][:100]}")
    
    return chunks_data

def getChunkText(chunks_data):
    """
    Docstring for getChunkText
    
    :param chunks_data: lis/dictionary of all chunks extracted form json file
    :return: Returns a list of text chunks or content of the chunks
    :rtype: list
    """
    chunks_text = []
    for chunk_test in chunks_data:
        chunks_text.append(chunk_test['text'])
    return chunks_text

def generateEmbedding(chunks_data, model_name='all-mpnet-base-v2', batch_size = 32):
    """
    Docstring for generateEmbedding
    
    :param chunks_data: List/dictionary of all chunks from the json file
    :param model_name: 'all-mpnet-base-v2'
    :param batch_size: 32
    """

    # Loading the model
    print(f"\nStep 1: Loaded model '{model_name}'....")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"    ✓ Model loaded successfully")
    print(f"    Embedding dimensions: {embedding_dim}")

    #Extract the text from chunks
    print(f"\nStep 2: Extracting text from {len(chunks_data)} chunks.....")
    texts = getChunkText(chunks_data)
    print(f"    ✓ Text extracted")

    #Generate the embeddings with batch processing
    print(f"\nStep 3: Generating embeddings (batch_size={batch_size} chunks....)")
    print(f"  This will process {len(texts)} chunks in {(len(texts) + batch_size - 1) // batch_size} batches")

    start_time = datetime.now()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True #Normalise for cosine similirity
    )
    end_time = datetime.now()
    duration = (end_time-start_time).total_seconds()

    print(f"    ✓ Embeddings generated in {duration:.2f} seconds")
    print(f"    Average: {duration/len(texts):.3f} seconds per chunk")
    print(f"    Throughput: {len(texts)/duration:.1f} chunks/second")

    # Verify normalisation 
    norms = np.linalg.norm(embeddings, axis = 1)
    is_normalised = np.allclose(norms, 1.0, atol=1e-5)
    print(f"    Normalised: {is_normalised} (all vectors have length ≈ 1.0))")
    return {
        'embeddings': embeddings,
        'model_name': model_name,
        'embedding_dim': embedding_dim,
        'num_chunks': len(texts),
        'batch_size': batch_size,
        'generation_time': duration,
        'normalised': is_normalised
    }

def save_embeddings(chunks_data, embeddings_info, output_dir="data/embeddings"):
    """
    Docstring for save_embeddings - saving embeddings and metadata to disk
    
    :param chunks_data: Original chunk data
    :param embeddings_info: Dictionary with embeddings and metadata
    :param output_dir:  Directory to save files
    """""

    print(f"\n{'='*70}")
    print(f"SAVING EMBEDDINGS")
    print(f"{'='*70}\n")

    os.makedirs(output_dir, exist_ok=True)
    print(f"    ✓ Created directory: {output_dir}")

    embeddings_file = os.path.join(output_dir, "embeddings.npz")
    np.savez_compressed(
        embeddings_file,
        embeddings=embeddings_info['embeddings'],
        model_name=embeddings_info['model_name'],
        embedding_dim=embeddings_info['embedding_dim'],
        num_chunks=embeddings_info['num_chunks'],
        generation_time=embeddings_info['generation_time']
    )
    print(f"    ✓ Embeddings saved to: {embeddings_file}")
    
    # Save chunks metadata
    metadata_file = os.path.join(output_dir, "chunks_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Metadata saved to: {metadata_file}")

    # Save configuration/statistics
    stats_file = os.path.join(output_dir, "embedding_stats.json")
    stats = {
        'model_name': embeddings_info['model_name'],
        'embedding_dimensions': embeddings_info['embedding_dim'],
        'total_chunks': embeddings_info['num_chunks'],
        'batch_size': embeddings_info['batch_size'],
        'generation_time_seconds': embeddings_info['generation_time'],
        'normalised': embeddings_info['normalised'],
        'file_size_bytes': embeddings_info['embeddings'].nbytes,
        'file_size_mb': embeddings_info['embeddings'].nbytes / (1024 * 1024),
        'timestamp': datetime.now().isoformat()
    }

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to: {stats_file}")
    
    return {
        'embeddings_file': embeddings_file,
        'metadata_file': metadata_file,
        'stats_file': stats_file,
        'stats': stats
    }

def display_summary(embeddings_info, save_info):
    """
    Docstring for display_summary which displays comprehensive summary of embeding generation
    
    :param embeddings_info: list of embeddings and their metadata
    :param save_info: where
    """""
    print(f"\n{'='*70}")
    print(f"EMBEDDING GENERATION SUMMARY")
    print(f"{'='*70}\n")

    stats = save_info['stats']
    
    print(f"Model Information:")
    print(f"  Model: {stats['model_name']}")
    print(f"  Embedding dimensions: {stats['embedding_dimensions']}")
    print(f"  Normalized: {stats['normalised']}")
    print()

    print(f"Processing Statistics:")
    print(f"  Total chunks processed: {stats['total_chunks']}")
    print(f"  Batch size: {stats['batch_size']}")
    print(f"  Generation time: {stats['generation_time_seconds']:.2f} seconds")
    print(f"  Throughput: {stats['total_chunks']/stats['generation_time_seconds']:.1f} chunks/second")
    print()

    # Validation
    print(f"Validation:")
    if stats['total_chunks'] >= 200:
        print(f"  ✅ PASSED: {stats['total_chunks']} chunks (requirement: 200+)")
    else:
        print(f"  ❌ FAILED: {stats['total_chunks']} chunks (requirement: 200+)")
    
    if stats['normalised']:
        print(f"  ✅ PASSED: Vectors normalized for cosine similarity")
    else:
        print(f"  ⚠️  WARNING: Vectors not normalized")


def main():
    """
    Main execution function for Part 3. 
    """
    print("=" * 70)
    print("PART 3: EMBEDDING GENERATION")
    print("=" * 70)

    print(f"    Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Step 1: Load chunks from Part 2
        chunks_data = load_chunks("data/processed/all_chunks.json")

        #Step 2: Generating embeddings
        embeddings_info = generateEmbedding(
            chunks_data,
            model_name='all-mpnet-base-v2',
            batch_size=32
        )

        #Step 3: Save embeddings
        save_info = save_embeddings(chunks_data, embeddings_info)

        display_summary(embeddings_info, save_info)

        print()
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return embeddings_info, save_info
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please ensure Part 2 has been completed.")
        raise

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        raise



if __name__ == '__main__':
    main()