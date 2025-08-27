# ============================================================================
# SECTION 1: ALL IMPORTS AT THE TOP
# ============================================================================

# --- Core Python Libraries ---
import os
import re
import json
import time
import sqlite3
import tempfile
import base64
import subprocess
import io
from io import BytesIO
from collections import Counter
import clip
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import math
from PIL import Image, ImageEnhance
# --- AI & Data Processing Libraries ---
import torch
import numpy as np
import faiss
import fitz  # PyMuPDF
import cv2
import pytesseract
import networkx as nx
import librosa
from PIL import Image
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download

# --- NLTK Imports ---
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# --- UI Import ---
import gradio as gr


# ============================================================================
# SECTION 2: GLOBAL SETUP & MODEL LOADING
# ============================================================================

# --- Global Variables ---
# We define them here, but they will be initialized inside the main() function
DB_PATH = "knowledge_base.db"
FAISS_INDEX_PATH = "faiss_index.idx"
ID_MAP_PATH = "id_map.json"
faiss_index = None
sentence_id_map = {}
embedding_model = None
llm_model = None
whisper_processor = None
whisper_model = None
knowledge_graph = None
STOPWORDS = None # Will be initialized in main()

# --- NLTK Data Setup ---
def setup_nltk():
    """Checks for and downloads necessary NLTK data packages."""
    try:
        # Check if packages are already downloaded to avoid unnecessary calls
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("📥 Downloading necessary NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)

# --- Model & Asset Downloader ---
def setup_models_and_assets():
    """
    Downloads all necessary AI models and Piper voice files if they don't exist locally.
    This function should be called once when the application starts.
    """
    global llm_model, whisper_processor, whisper_model, embedding_model
    
    # --- Download Piper Voice Model ---
    voice_dir = "piper_voices/en_US"
    os.makedirs(voice_dir, exist_ok=True)
    onnx_path = os.path.join(voice_dir, "voice.onnx")
    json_path = os.path.join(voice_dir, "voice.json")

    if not os.path.exists(onnx_path) or not os.path.exists(json_path):
        print("🗣️ Downloading Piper voice model...")
        repo_id = "rhasspy/piper-voices"
        onnx_file = "en/en_US/kusal/medium/en_US-kusal-medium.onnx"
        json_file = "en/en_US/kusal/medium/en_US-kusal-medium.onnx.json"
        
        hf_hub_download(repo_id=repo_id, filename=onnx_file, local_dir=voice_dir, local_dir_use_symlinks=False)
        hf_hub_download(repo_id=repo_id, filename=json_file, local_dir=voice_dir, local_dir_use_symlinks=False)
        
        # Rename files for consistent pathing in the app
        os.rename(os.path.join(voice_dir, onnx_file), onnx_path)
        os.rename(os.path.join(voice_dir, json_file), json_path)
        print("✅ Piper voice downloaded.")
    
    # --- Load All AI Models ---
    print("🧠 Loading base models...")
    try:
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ Hardware check: Using '{device}' for compatible models.")
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        llm_model = GPT4All("Llama-3.2-1B-Instruct-Q4_0.gguf", device=device)
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        
        print("✅ Successfully loaded base models")
    except Exception as e:
        print(f"❌ Failed to load models: {str(e)}")
        raise

# ============================================================================
# SECTION: DATABASE AND GRAPH FUNCTIONS (SQLite + NetworkX) - COMPLETE
# ============================================================================

def initialize_database():
    """
    Creates the SQLite database and all necessary tables with their full schemas.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Create the full table for paragraphs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paragraphs (
                id TEXT PRIMARY KEY,
                text TEXT,
                header TEXT,
                page_num INTEGER,
                source_pdf TEXT,
                bbox_range TEXT,
                tags TEXT,
                full_page_ocr TEXT
            )
        ''')

        # Create the full table for images
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                caption TEXT,
                ocr_text TEXT,
                data TEXT,
                page_num INTEGER,
                source_pdf TEXT,
                bbox TEXT,
                tags TEXT
            )
        ''')

        # Create the table for all relationships (edges)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                source_id TEXT,
                target_id TEXT,
                type TEXT,
                PRIMARY KEY (source_id, target_id, type)
            )
        ''')

        conn.commit()

def add_paragraph_to_db(cursor, props):
    # Add the new property to the INSERT statement
    cursor.execute("INSERT OR REPLACE INTO paragraphs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
        props['id'], props['text'], props['header'], props['page_num'],
        props['source_pdf'], str(props.get('bbox_range')),
        json.dumps(props.get('tags', [])), props.get('full_page_ocr', '')
    ))

def add_image_to_db(cursor, props):
    """
    Inserts an image's data into the SQLite database, now with the corrected
    json.dumps for tags.
    """
    cursor.execute("INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
        props['id'],
        props['caption'],
        props['ocr_text'],
        props['data'],
        props['page_num'],
        props['source_pdf'],
        str(props.get('bbox')),
        json.dumps(props.get('tags', []))  # <-- THIS IS THE FIX
    ))

# --- NEW FUNCTIONS TO HANDLE ALL RELATIONSHIPS ---
def add_relationship(cursor, source_id, target_id, rel_type):
    """Inserts a generic relationship into the database."""
    cursor.execute("INSERT OR IGNORE INTO relationships VALUES (?, ?, ?)", (
        source_id, target_id, rel_type
    ))

def add_tags_and_relationships(cursor, item_id, item_tags):
    """Adds tag relationships for a given paragraph or image."""
    for tag in item_tags:
        # The target_id for a tag is the tag name itself for simplicity
        add_relationship(cursor, item_id, tag, 'HAS_TAG')



def load_graph_from_db(target_pdf):
    """
    Loads nodes and edges from the SQLite DB for a specific PDF
    and constructs an in-memory NetworkX graph. (Corrected Version)
    """
    if not os.path.exists(DB_PATH):
        return None

    G = nx.Graph()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Get all paragraphs for the target PDF and add them as nodes
        for row in cursor.execute("SELECT id, tags FROM paragraphs WHERE source_pdf=?", (target_pdf,)):
            # --- THIS IS THE FIX ---
            # Use the correct index (row[1]) for the tags column.
            G.add_node(row[0], type='paragraph', tags=json.loads(row[1]))
            # --------------------

        # Get all images for the target PDF and add them as nodes
        for row in cursor.execute("SELECT id, tags FROM images WHERE source_pdf=?", (target_pdf,)):
            # --- THIS IS THE FIX ---
            # Use the correct index (row[1]) for the tags column.
            G.add_node(row[0], type='image', tags=json.loads(row[1]))
            # --------------------

        # Get all relationships for the nodes in the current graph
        nodes_in_graph = list(G.nodes())
        if not nodes_in_graph: return G # Return empty graph if no nodes

        placeholders = ', '.join(['?'] * len(nodes_in_graph))
        query = f"SELECT source_id, target_id FROM relationships WHERE source_id IN ({placeholders})"

        for row in cursor.execute(query, nodes_in_graph):
            if row[1] in nodes_in_graph:
                G.add_edge(row[0], row[1])

    return G

def cleanup_database():
    """Cleans up the SQLite database file."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    initialize_database() # Recreate the empty tables
    print("✅ Database cleaned up")


import os
import sqlite3

# This is a global constant you should have defined
DB_PATH = "knowledge_base.db"

def get_statistics():
    """
    Gets statistics about the processed knowledge base directly from the SQLite database.
    """
    # Check if the database file exists before trying to connect
    if not os.path.exists(DB_PATH):
        print("⚠️ Database file not found. Returning empty stats.")
        return {
            'paragraphs': 0,
            'images': 0,
            'tags': 0,
            'pdf_sources': [],
            'total_pages': 0
        }

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        stats = {}

        # Count total paragraphs
        stats['paragraphs'] = cursor.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]

        # Count total images
        stats['images'] = cursor.execute("SELECT COUNT(*) FROM images").fetchone()[0]

        # Count unique tags by querying the relationships table
        stats['tags'] = cursor.execute("SELECT COUNT(DISTINCT target_id) FROM relationships WHERE type='HAS_TAG'").fetchone()[0]

        # Get a list of all unique PDF sources
        stats['pdf_sources'] = [row[0] for row in cursor.execute("SELECT DISTINCT source_pdf FROM paragraphs")]

        # Get the highest page number to calculate total pages across all documents
        max_page_row = cursor.execute("SELECT MAX(page_num) FROM paragraphs").fetchone()
        max_page = max_page_row[0] if max_page_row and max_page_row[0] is not None else -1
        stats['total_pages'] = max_page + 1

        return stats
# **ENHANCED CHUNKING SYSTEM**

class HybridChunker:
    """
    Combines EnhancedChunker with fallback sentence-based chunking
    """
    def __init__(self):
        self.enhanced_chunker = EnhancedChunker()
        self.min_sections_threshold = 2  # Minimum sections to use enhanced chunking
        
    def process_page_intelligently(self, page, page_num, pdf_path, full_page_ocr_text):
        """
        Intelligently chooses between enhanced and simple chunking based on structure
        """
        # First, try enhanced chunking
        enhanced_blocks = self.enhanced_chunker.extract_enhanced_blocks(page)
        sections = self.enhanced_chunker.group_into_sections(enhanced_blocks)
        
        # Check if we found meaningful structure
        if len(sections) >= self.min_sections_threshold and any(s['header'] for s in sections):
            print(f"📊 Using enhanced chunking for page {page_num + 1}")
            return self.enhanced_chunker.create_section_chunks(
                sections, page_num, pdf_path, full_page_ocr_text
            )
        else:
            print(f"📝 Using fallback sentence chunking for page {page_num + 1}")
            return self._fallback_sentence_chunking(page, page_num, pdf_path, full_page_ocr_text)
    
    def _fallback_sentence_chunking(self, page, page_num, pdf_path, full_page_ocr_text):
        """
        Fallback to simple sentence-based chunking for unstructured content
        """
        text = page.get_text()
        if not text.strip():
            return []
        
        # Clean and split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        max_words = 150  # Target chunk size
        
        for i, sentence in enumerate(sentences):
            words_in_sentence = len(sentence.split())
            
            if current_word_count + words_in_sentence <= max_words or not current_chunk:
                current_chunk.append(sentence)
                current_word_count += words_in_sentence
            else:
                # Create chunk from current sentences
                chunk_text = '. '.join(current_chunk) + '.'
                chunk_id = f"fallback_{page_num}_{len(chunks)}"
                
                # Extract tags using the enhanced method
                tags = self.enhanced_chunker.extract_enhanced_tags(chunk_text, "")
                
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'header': f"Section {len(chunks) + 1}",
                    'source_pdf': pdf_path,
                    'page_num': page_num,
                    'full_page_ocr': full_page_ocr_text,
                    'section_type': 'paragraph',
                    'tags': tags,
                    'bbox_range': (0, 0)  # Default bbox for fallback chunks
                })
                
                # Start new chunk
                current_chunk = [sentence]
                current_word_count = words_in_sentence
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunk_id = f"fallback_{page_num}_{len(chunks)}"
            tags = self.enhanced_chunker.extract_enhanced_tags(chunk_text, "")
            
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'header': f"Section {len(chunks) + 1}",
                'source_pdf': pdf_path,
                'page_num': page_num,
                'full_page_ocr': full_page_ocr_text,
                'section_type': 'paragraph',
                'tags': tags,
                'bbox_range': (0, 0)
            })
        
        return chunks



class EnhancedChunker:

    def __init__(self):
        self.section_patterns = [
            r'^\d+\.\d+\.\d+\s+[A-Z]',  # 3.3.1 Test for starch
            r'^\d+\.\d+\s+[A-Z]',       # 3.3 How to Test Different Components
            r'^\d+\s+[A-Z]',            # 3 CHAPTER
            r'^[A-Z][A-Z\s]+$',         # ALL CAPS HEADERS
            r'^Activity\s+\d+\.\d+',    # Activity 3.5
            r'^Fig\.\s+\d+\.\d+',       # Fig. 3.7
            r'^Table\s+\d+\.\d+',       # Table 3.3
        ]

        self.subsection_patterns = [
            r'^\s*•\s+',                # Bullet points
            r'^\s*\d+\.\s+',            # Numbered lists
            r'^\s*\([a-z]\)',           # (a) (b) (c)
            r'^\s*\u2022\s+',           # Unicode bullet
        ]

    def is_section_header(self, text):
        """Detect if text block is a section header"""
        text = text.strip()
        if len(text) < 3 or len(text) > 200:
            return False

        for pattern in self.section_patterns:
            if re.match(pattern, text):
                return True
        return False

    def is_subsection(self, text):
        """Detect if text block is a subsection"""
        text = text.strip()
        for pattern in self.subsection_patterns:
            if re.match(pattern, text):
                return True
        return False

    def extract_enhanced_blocks(self, page):
        """Extract text blocks with enhanced structure detection"""
        # Get text blocks with position info
        blocks = page.get_text("blocks")
        enhanced_blocks = []

        for block in blocks:
            if len(block) < 5:
                continue

            x0, y0, x1, y1, text = block[:5]
            text = text.strip()

            if not text or len(text) < 3:
                continue

            # Determine block type
            block_type = "paragraph"
            if self.is_section_header(text):
                block_type = "section_header"
            elif self.is_subsection(text):
                block_type = "subsection"
            elif re.match(r'^Fig\.\s+\d+\.\d+', text):
                block_type = "figure_caption"
            elif re.match(r'^Table\s+\d+\.\d+', text):
                block_type = "table_caption"

            enhanced_blocks.append({
                'bbox': (x0, y0, x1, y1),
                'text': text,
                'type': block_type,
                'y_position': y0
            })

        # Sort by y-position (top to bottom)
        enhanced_blocks.sort(key=lambda x: x['y_position'])
        return enhanced_blocks

    def group_into_sections(self, enhanced_blocks):
        """Group blocks into semantic sections"""
        sections = []
        current_section = {
            'header': '',
            'content': [],
            'type': 'section',
            'start_y': 0,
            'end_y': 0
        }

        for block in enhanced_blocks:
            if block['type'] == 'section_header':
                # Start new section
                if current_section['content'] or current_section['header']:
                    sections.append(current_section)

                current_section = {
                    'header': block['text'],
                    'content': [],
                    'type': 'section',
                    'start_y': block['y_position'],
                    'end_y': block['y_position']
                }
            else:
                # Add to current section
                current_section['content'].append(block['text'])
                current_section['end_y'] = block['y_position']

        # Add final section
        if current_section['content'] or current_section['header']:
            sections.append(current_section)

        return sections



    def create_section_chunks(self, sections, page_num, pdf_path, full_page_ocr_text):
        """Create optimized chunks from sections"""
        chunks = []

        for idx, section in enumerate(sections):
            # Combine header and content
            full_text = section['header']
            if section['content']:
                full_text += '\n' + '\n'.join(section['content'])

            # Clean text
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            # Skip very short sections
            if len(full_text.split()) < 10:
                continue

            chunk_id = f"section_{page_num}_{idx}"

            # Enhanced tag extraction
            tags = self.extract_enhanced_tags(full_text, section['header'])

            chunks.append({
                'id': chunk_id,
                'text': full_text,
                'header': section['header'],
                'source_pdf': pdf_path,
                'page_num': page_num,
                'full_page_ocr': full_page_ocr_text, # <-- Add this new field
                'section_type': section['type'],
                'tags': tags,
                'bbox_range': (section['start_y'], section['end_y'])
            })

        return chunks

    def extract_enhanced_tags(self, text, header):
        """Extract meaningful tags from text and header"""
        # Combine header and text for tag extraction
        combined_text = header + ' ' + text

        # Scientific/educational terms patterns
        scientific_patterns = [
            r'\b[A-Z][a-z]+(?:ose|ase|ine|ate|ide)\b',  # Chemical suffixes
            r'\b(?:test|experiment|activity|method|procedure)\b',  # Process terms
            r'\b(?:starch|protein|carbohydrate|fat|glucose|enzyme)\b',  # Biological terms
        ]

        # Extract pattern-based tags
        pattern_tags = []
        for pattern in scientific_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            pattern_tags.extend([m.lower() for m in matches])

        # Extract noun-based tags
        tokens = word_tokenize(combined_text.lower())
        tagged = pos_tag(tokens)
        nouns = [word for word, tag in tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]

        # Filter and combine tags
        all_tags = pattern_tags + nouns  # <-- Corrected line

        filtered_tags = [
            tag for tag in all_tags
            if tag not in STOPWORDS
            and len(tag) > 2
            and tag not in ['image', 'content', 'figure', 'table', 'page']
        ]

        # Return top 5 most frequent tags
        tag_counts = Counter(filtered_tags)
        return [tag for tag, _ in tag_counts.most_common(5)]

# **ENHANCED IMAGE DETECTION SYSTEM**

class EnhancedImageDetector:
    """Enhanced image detection to prevent fragmentation"""

    def __init__(self):
        self.min_image_size = (80, 80)  # Minimum size for valid images
        self.merge_threshold = 50  # Pixels for merging nearby images

    def detect_image_regions(self, page):
        """Detect image regions with improved accuracy"""
        # Get high-resolution page rendering
        mat = fitz.Matrix(2, 2)  # 2x scaling
        pix = page.get_pixmap(matrix=mat)

        # Add check to ensure pix is not None
        if pix is None:
            print(f"⚠️ Skipping image detection on page due to missing pixmap.")
            return []

        # --- Start of fix ---

        # 1. Use pix.samples to get raw pixel data, not pix.tobytes()
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)

        # 2. Reshape using the actual number of channels (pix.n)
        if pix.n * pix.width * pix.height != len(pix.samples):
            print(f"⚠️ Skipping image detection on page {page.number} due to data mismatch.")
            return []
        img_array = img_array.reshape(pix.height, pix.width, pix.n)

        # 3. Handle different image formats (like RGBA or Grayscale)
        if pix.n == 4:  # RGBA image
            # Convert to RGB for OpenCV processing
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:  # Grayscale image
            # Convert to RGB for consistency
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        # --- End of fix ---

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Enhanced edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Morphological operations to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and merge contours
        image_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Scale back to original coordinates
            x, y, w, h = x//2, y//2, w//2, h//2

            # Filter by size
            if w >= self.min_image_size[0] and h >= self.min_image_size[1]:
                image_regions.append((x, y, x+w, y+h))

        # Merge overlapping regions
        merged_regions = self.merge_overlapping_regions(image_regions)

        return merged_regions

    def merge_overlapping_regions(self, regions):
        """Merge overlapping or nearby image regions"""
        if not regions:
            return []

        merged = []
        regions = sorted(regions, key=lambda r: (r[1], r[0]))  # Sort by y, then x

        for region in regions:
            if not merged:
                merged.append(region)
                continue

            # Check if current region should be merged with any existing
            merged_with_existing = False
            for i, existing in enumerate(merged):
                if self.should_merge_regions(existing, region):
                    # Merge regions
                    merged[i] = (
                        min(existing[0], region[0]),  # min x
                        min(existing[1], region[1]),  # min y
                        max(existing[2], region[2]),  # max x
                        max(existing[3], region[3])   # max y
                    )
                    merged_with_existing = True
                    break

            if not merged_with_existing:
                merged.append(region)

        return merged

    def should_merge_regions(self, region1, region2):
        """Check if two regions should be merged"""
        x1, y1, x2, y2 = region1
        x3, y3, x4, y4 = region2

        # Calculate distances
        h_distance = min(abs(x2 - x3), abs(x4 - x1))
        v_distance = min(abs(y2 - y3), abs(y4 - y1))

        # Check for overlap or proximity
        h_overlap = max(0, min(x2, x4) - max(x1, x3))
        v_overlap = max(0, min(y2, y4) - max(y1, y3))

        return (h_overlap > 0 and v_overlap > 0) or (h_distance < self.merge_threshold and v_distance < self.merge_threshold)

    # In the EnhancedImageDetector class

    def extract_image_from_region(self, page, region, padding=20):
        """Extract image from specific region with padding to include labels."""
        x1, y1, x2, y2 = region

        # Add padding to the bounding box, ensuring it stays within page limits
        page_rect = page.rect
        x1 = max(page_rect.x0, x1 - padding)
        y1 = max(page_rect.y0, y1 - padding)
        x2 = min(page_rect.x1, x2 + padding)
        y2 = min(page_rect.y1, y2 + padding)

        # Create clip rectangle from the new, larger coordinates
        clip_rect = fitz.Rect(x1, y1, x2, y2)

        # Render the clipped region
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)

        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))

        return img, img_data

    def get_surrounding_text(self, page, region, margin=100):
        """Get text surrounding an image region with improved accuracy"""
        x1, y1, x2, y2 = region

        # Expand region to include surrounding text
        expanded_rect = fitz.Rect(
            max(0, x1 - margin),
            max(0, y1 - margin),
            min(page.rect.width, x2 + margin),
            min(page.rect.height, y2 + margin)
        )

        # Get text blocks in expanded region
        blocks = page.get_text("blocks", clip=expanded_rect)

        surrounding_texts = []
        for block in blocks:
            if len(block) < 5:
                continue

            bx0, by0, bx1, by1, text = block[:5]
            text = text.strip()

            if not text:
                continue

            # Check if text block is actually surrounding the image (not inside it)
            if not (bx0 >= x1 and by0 >= y1 and bx1 <= x2 and by1 <= y2):
                surrounding_texts.append(text)

        return surrounding_texts

def get_document_summary_context(pdf_path):
    """
    Finds summary sections by searching both structured headers AND the full-page OCR text.
    """
    if not os.path.exists(DB_PATH):
        return None
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # This query now checks the header OR the full page OCR text for summary keywords
        sql_query = """
            SELECT text FROM paragraphs
            WHERE source_pdf = ? AND (
                LOWER(header) LIKE '%summary%' OR
                LOWER(header) LIKE '%conclusion%' OR
                LOWER(header) LIKE '%abstract%' OR
                full_page_ocr LIKE '%summary%'  -- Search the new OCR layer
            )
            LIMIT 5
        """
        # Note: You must add a 'full_page_ocr' TEXT column to your 'paragraphs' table in initialize_database()

        results = cursor.execute(sql_query, (pdf_path,)).fetchall()
        return "\n\n".join([row[0] for row in results]) if results else None


# **ENHANCED MAIN PROCESSING FUNCTIONS**

def extract_enhanced_ocr(image):
    """Enhanced OCR with preprocessing"""
    if image is None:
        return ""

    try:
        # Convert PIL image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # OCR with custom config
        custom_config = r'--oem 3 --psm 6'
        ocr_text = pytesseract.image_to_string(thresh, config=custom_config)

        return ocr_text.strip()
    except Exception as e:
        print(f"⚠️ Enhanced OCR failed: {str(e)}")
        return ""


def generate_enhanced_tags(ocr_text, surrounding_text, image_caption):
    """Generate enhanced tags from multiple sources"""
    # Combine all text sources
    all_text = f"{ocr_text} {' '.join(surrounding_text)} {image_caption}"

    # Extract scientific/educational terms
    scientific_terms = re.findall(r'\b(?:starch|protein|carbohydrate|fat|glucose|enzyme|test|experiment|activity|method|procedure|molecule|acid|base|solution|reaction|process|function|structure|system|component|nutrient|vitamin|mineral|fiber|energy|metabolism|digestion|absorption|synthesis|analysis|observation|result|conclusion|hypothesis|theory|principle|law|formula|equation|calculation|measurement|unit|scale|graph|chart|diagram|illustration|figure|table|data|sample|specimen|culture|organism|cell|tissue|organ|blood|urine|saliva|enzyme|catalyst|indicator|reagent|chemical|compound|mixture|solution|concentration|dilution|filtration|precipitation|crystallization|distillation|extraction|purification|identification|quantification|qualitative|quantitative)\b', all_text, re.IGNORECASE)

    # Extract noun phrases
    tokens = word_tokenize(all_text.lower())
    tagged_tokens = pos_tag(tokens)
    nouns = [word for word, tag in tagged_tokens if tag in ['NN', 'NNS', 'NNP', 'NNPS']]

    # Combine and filter
    all_tags = scientific_terms + nouns
    filtered_tags = [
        tag.lower() for tag in all_tags
        if tag.lower() not in STOPWORDS
        and len(tag) > 2
        and tag.lower() not in ['topic','separate','context','questions','fig','image', 'content', 'figure', 'table', 'page', 'text', 'item', 'thing', 'part', 'way', 'time', 'place', 'person', 'people', 'man', 'woman', 'child', 'day', 'year', 'work', 'life', 'home', 'world', 'country', 'state', 'city', 'school', 'group', 'example', 'problem', 'question', 'answer', 'right', 'left', 'side', 'end', 'beginning', 'middle', 'top', 'bottom', 'front', 'back', 'inside', 'outside', 'above', 'below', 'over', 'under', 'between', 'among', 'through', 'across', 'around', 'near', 'far', 'close', 'open', 'shut', 'good', 'bad', 'big', 'small', 'long', 'short', 'high', 'low', 'hot', 'cold', 'warm', 'cool', 'new', 'old', 'young', 'great', 'little', 'own', 'other', 'last', 'first', 'next', 'few', 'many', 'much', 'most', 'some', 'all', 'any', 'no', 'not', 'only', 'very', 'well', 'still', 'just', 'now', 'here', 'there', 'then', 'when', 'where', 'how', 'what', 'which', 'who', 'why', 'this', 'that', 'these', 'those', 'such', 'same', 'different', 'another', 'each', 'every', 'both', 'either', 'neither', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    ]

    # Return top 5 unique tags
    tag_counts = Counter(filtered_tags)
    return list(dict.fromkeys([tag for tag, _ in tag_counts.most_common(5)]))


class CLIPImageRetriever:
    """
    Advanced image retrieval using CLIP embeddings with multiple search strategies
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖼️ Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_index = None
        self.image_metadata = {}
        
    def build_clip_index(self, images_data):
        """
        Build FAISS index for CLIP image embeddings
        """
        print(f"🔍 Building CLIP index for {len(images_data)} images...")
        
        embeddings = []
        metadata = {}
        
        for idx, img_data in enumerate(images_data):
            try:
                # Decode image from base64
                img_bytes = base64.b64decode(img_data['data'])
                image = Image.open(BytesIO(img_bytes))
                
                # Preprocess and get CLIP embedding
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_embedding = self.model.encode_image(image_input)
                    image_embedding = F.normalize(image_embedding, p=2, dim=1)
                
                embeddings.append(image_embedding.cpu().numpy().flatten())
                metadata[idx] = {
                    'id': img_data['id'],
                    'caption': img_data['caption'],
                    'page_num': img_data['page_num'],
                    'source_pdf': img_data['source_pdf'],
                    'tags': img_data.get('tags', []),
                    'data': img_data['data']
                }
                
            except Exception as e:
                print(f"⚠️ Error processing image {idx}: {str(e)}")
                continue
        
        if embeddings:
            # Create FAISS index
            embedding_array = np.array(embeddings, dtype=np.float32)
            embedding_dim = embedding_array.shape[1]
            self.clip_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for normalized vectors
            self.clip_index.add(embedding_array)
            self.image_metadata = metadata
            print(f"✅ CLIP index built with {len(embeddings)} images")
        
        return self.clip_index is not None
    
    def search_images_by_text(self, query_text, k=5):
        """
        Search images using CLIP text embeddings (Stage 1: Visual Search)
        """
        if self.clip_index is None:
            return []
        
        # Combine query and answer for comprehensive search
        combined_text = f"{query_text}".strip()
        
        # Get text embedding
        text_tokens = clip.tokenize([combined_text]).to(self.device)
        
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = F.normalize(text_embedding, p=2, dim=1)
        
        # Search in CLIP index
        query_embedding = text_embedding.cpu().numpy()
        scores, indices = self.clip_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.image_metadata):
                img_data = self.image_metadata[idx].copy()
                img_data['clip_score'] = float(score)
                results.append(img_data)
        
        return results
    
    def search_images_by_page_context(self, relevant_paragraph_ids, target_pdf):
        """
        Find images on the same pages as relevant paragraphs (Stage 2: Contextual Search)
        """
        if not os.path.exists(DB_PATH):
            return []
        
        # Get page numbers for relevant paragraphs
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Get pages of relevant paragraphs
            placeholders = ', '.join(['?'] * len(relevant_paragraph_ids))
            page_query = f"SELECT DISTINCT page_num FROM paragraphs WHERE id IN ({placeholders}) AND source_pdf = ?"
            params = relevant_paragraph_ids + [target_pdf]
            relevant_pages = [row[0] for row in cursor.execute(page_query, params).fetchall()]
            
            if not relevant_pages:
                return []
            
            # Get all images from these pages
            page_placeholders = ', '.join(['?'] * len(relevant_pages))
            img_query = f"SELECT * FROM images WHERE page_num IN ({page_placeholders}) AND source_pdf = ?"
            img_params = relevant_pages + [target_pdf]
            
            results = cursor.execute(img_query, img_params).fetchall()
            column_names = [description[0] for description in cursor.description]
            
            contextual_images = []
            for row in results:
                img_dict = dict(zip(column_names, row))
                img_dict['context_score'] = 1.0  # High score for contextual relevance
                contextual_images.append(img_dict)
            
            return contextual_images


class HybridKnowledgeStorage:
    """
    Enhanced storage system with dual FAISS indexes and knowledge graph
    """
    def __init__(self):
        self.text_faiss_index = None
        self.clip_faiss_index = None
        self.text_id_map = {}
        self.clip_retriever = CLIPImageRetriever()
        self.knowledge_graph = None
    
    def build_hybrid_indexes(self, paragraphs_data, images_data):
        """
        Build both text and image FAISS indexes
        """
        # Build text FAISS index (existing functionality enhanced)
        self.text_faiss_index, self.text_id_map = self._build_text_faiss(paragraphs_data)
        
        # Build CLIP image index
        self.clip_retriever.build_clip_index(images_data)
        
        print("✅ Hybrid knowledge storage indexes built successfully")
    
    def _build_text_faiss(self, paragraphs_data):
        """
        Build the text FAISS index with memory optimization
        """
        if not paragraphs_data:
            return None, {}
        
        try:
            batch_size = 32
            embeddings = []
            
            print(f"📊 Building text FAISS index for {len(paragraphs_data)} paragraphs...")
            
            for i in range(0, len(paragraphs_data), batch_size):
                batch = paragraphs_data[i:i + batch_size]
                texts = [p['text'] for p in batch]
                
                batch_embeddings = embedding_model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=16,
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings)
            
            # Build FAISS index
            embedding_array = np.array(embeddings, dtype=np.float32)
            embedding_dim = embedding_array.shape[1]
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(embedding_array)
            
            # Create ID mapping
            id_map = {
                i: {
                    "paragraph_id": p["id"],
                    "embedding_id": f"embed_{p['id']}",
                    "text": p["text"],
                    "tags": p.get("tags", [])
                }
                for i, p in enumerate(paragraphs_data)
            }
            
            print(f"✅ Text FAISS index built with {index.ntotal} paragraphs")
            return index, id_map
            
        except Exception as e:
            print(f"❌ Text FAISS index build failed: {str(e)}")
            return None, {}



class MMRRanker:
    """
    Maximal Marginal Relevance implementation for diverse document selection
    """
    def __init__(self, lambda_param=0.7):
        self.lambda_param = lambda_param  # Balance between relevance and diversity
    
    def calculate_mmr(self, query_embedding, document_embeddings, document_ids, k=5):
        """
        Calculate MMR to select diverse and relevant documents
        """
        if len(document_embeddings) == 0:
            return []
        
        # Calculate relevance scores (cosine similarity with query)
        relevance_scores = cosine_similarity([query_embedding], document_embeddings)[0]
        
        selected_indices = []
        remaining_indices = list(range(len(document_embeddings)))
        
        # Select first document with highest relevance
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining documents using MMR formula
        while len(selected_indices) < min(k, len(document_embeddings)) and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]
                
                # Diversity component (maximum similarity to already selected)
                if selected_indices:
                    selected_embeddings = [document_embeddings[i] for i in selected_indices]
                    similarity_to_selected = cosine_similarity(
                        [document_embeddings[idx]], selected_embeddings
                    )[0]
                    max_similarity = np.max(similarity_to_selected)
                else:
                    max_similarity = 0
                
                # MMR formula: λ * relevance - (1-λ) * diversity
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                mmr_scores.append((mmr_score, idx))
            
            # Select document with highest MMR score
            best_score, best_idx = max(mmr_scores)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return selected document IDs in order
        return [document_ids[i] for i in selected_indices]
    
    def rerank_search_results(self, query_embedding, search_results, k=5):
        """
        Rerank search results using MMR for diversity
        """
        if not search_results:
            return []
        
        # Extract embeddings and IDs from search results
        embeddings = []
        doc_ids = []
        
        for result in search_results:
            # Assuming search results have embeddings or we can get them
            if 'embedding' in result:
                embeddings.append(result['embedding'])
                doc_ids.append(result['id'])
            else:
                # Generate embedding on the fly if needed
                text = result.get('text', result.get('caption', ''))
                if text:
                    emb = embedding_model.encode([text], normalize_embeddings=True)[0]
                    embeddings.append(emb)
                    doc_ids.append(result['id'])
        
        if not embeddings:
            return search_results[:k]
        
        # Apply MMR ranking
        selected_ids = self.calculate_mmr(query_embedding, embeddings, doc_ids, k)
        
        # Return results in MMR order
        reranked = []
        for doc_id in selected_ids:
            for result in search_results:
                if result.get('id') == doc_id:
                    reranked.append(result)
                    break
        
        return reranked


# ============================================================================
# INTEGRATION 5: ENHANCED MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_hybrid_pdfs(pdf_paths):
    """
    Enhanced PDF processing with hybrid chunking, dual indexing, and knowledge graph generation
    """
    global faiss_index, sentence_id_map
    
    hybrid_chunker = HybridChunker()
    image_detector = EnhancedImageDetector()
    hybrid_storage = HybridKnowledgeStorage()
    
    all_paragraphs, all_images = [], []
    seen_ids = set()
    pdf_processing_stats = {}
    
    print("📚 Starting enhanced hybrid PDF processing...")
    
    for pdf_idx, pdf_path in enumerate(pdf_paths):
        pdf_stats = {'paragraphs': 0, 'images': 0, 'pages': 0, 'kg_entities': 0}
        
        try:
            doc = fitz.open(pdf_path)
            pdf_name = os.path.basename(pdf_path)
            print(f"📘 Processing [{pdf_idx+1}/{len(pdf_paths)}]: {pdf_name} ({len(doc)} pages)")
            
            pdf_paragraphs = []  # Track paragraphs for this specific PDF
            pdf_images = []      # Track images for this specific PDF
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                if page is None:
                    continue
                    
                print(f"📄 Processing page {page_num + 1}/{len(doc)} of {pdf_name}")
                
                # Enhanced OCR with better preprocessing
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # Higher resolution
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    
                    # Improve OCR quality
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.2)
                    full_page_ocr_text = pytesseract.image_to_string(img, config='--psm 6').lower()
                    
                except Exception as e:
                    print(f"⚠️ OCR failed for page {page_num + 1}: {e}")
                    full_page_ocr_text = ""
                
                # HYBRID CHUNKING: Intelligent processing with better deduplication
                try:
                    chunks = hybrid_chunker.process_page_intelligently(
                        page, page_num, pdf_path, full_page_ocr_text
                    )
                    
                    # Enhanced deduplication with fuzzy matching
                    current_page_paragraphs = []
                    for chunk in chunks:
                        chunk_id = chunk.get('id', f"{pdf_path}_p{page_num}_{len(current_page_paragraphs)}")
                        
                        # Check for near-duplicates using text similarity
                        is_duplicate = False
                        chunk_text = chunk.get('text', '').strip().lower()
                        
                        if chunk_text and len(chunk_text) > 20:  # Skip very short chunks
                            for existing_para in all_paragraphs[-50:]:  # Check last 50 paragraphs
                                existing_text = existing_para.get('text', '').strip().lower()
                                # Simple similarity check (can be enhanced with edit distance)
                                if chunk_text in existing_text or existing_text in chunk_text:
                                    if len(chunk_text) > 0.8 * len(existing_text):
                                        is_duplicate = True
                                        break
                        
                        if not is_duplicate and chunk_id not in seen_ids:
                            chunk['source_pdf'] = pdf_path  # Ensure source tracking
                            chunk['pdf_index'] = pdf_idx
                            current_page_paragraphs.append(chunk)
                            pdf_paragraphs.append(chunk)
                            seen_ids.add(chunk_id)
                    
                    all_paragraphs.extend(current_page_paragraphs)
                    pdf_stats['paragraphs'] += len(current_page_paragraphs)
                    
                except Exception as e:
                    print(f"⚠️ Chunking failed for page {page_num + 1}: {e}")
                    continue
                
                # Enhanced image detection with better error handling
                try:
                    image_regions = image_detector.detect_image_regions(page)
                    print(f"🖼️ Detected {len(image_regions)} image regions on page {page_num + 1}")
                    
                    current_page_images = []
                    for idx, region in enumerate(image_regions):
                        try:
                            img, img_data = image_detector.extract_image_from_region(page, region)
                            
                            # Skip very small images
                            if img.size[0] < 50 or img.size[1] < 50:
                                continue
                                
                            # Enhanced OCR for images
                            ocr_text = extract_enhanced_ocr(img)
                            surrounding_text = image_detector.get_surrounding_text(page, region)
                            
                            # Better caption generation
                            if ocr_text and len(ocr_text.strip()) > 5:
                                caption = ocr_text[:150] + "..." if len(ocr_text) > 150 else ocr_text
                            elif surrounding_text:
                                caption = f"Figure from {' '.join(surrounding_text[:20])}"
                            else:
                                caption = f"Educational diagram from {pdf_name}, Page {page_num + 1}"
                            
                            # Enhanced tagging with domain-specific terms
                            combined_text = f"{caption} {' '.join(surrounding_text)} {ocr_text}"
                            tags = generate_enhanced_tags(combined_text, pdf_name, f"page_{page_num}")
                            
                            image_id = f"img_{pdf_idx}_{page_num}_{idx}"
                            image_data = {
                                'id': image_id,
                                'caption': caption,
                                'ocr_text': ocr_text,
                                'data': base64.b64encode(img_data).decode('utf-8'),
                                'source_pdf': pdf_path,
                                'page_num': page_num,
                                'tags': tags,
                                'bbox': region,
                                'surrounding_text': surrounding_text,
                                'pdf_index': pdf_idx,
                                'image_size': f"{img.size[0]}x{img.size[1]}"
                            }
                            
                            current_page_images.append(image_data)
                            pdf_images.append(image_data)
                            
                        except Exception as e:
                            print(f"⚠️ Error processing image {idx} on page {page_num + 1}: {str(e)}")
                    
                    all_images.extend(current_page_images)
                    pdf_stats['images'] += len(current_page_images)
                    
                except Exception as e:
                    print(f"⚠️ Image detection failed for page {page_num + 1}: {e}")
                
                # Enhanced database storage with batch operations
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        cursor = conn.cursor()
                        
                        # Batch insert paragraphs
                        for para in current_page_paragraphs:
                            add_paragraph_to_db(cursor, para)
                            add_relationship(cursor, para['id'], para['source_pdf'], 'PART_OF')
                            add_tags_and_relationships(cursor, para['id'], para.get('tags', []))
                        
                        # Batch insert images
                        for img_node in current_page_images:
                            add_image_to_db(cursor, img_node)
                            add_relationship(cursor, img_node['id'], img_node['source_pdf'], 'PART_OF')
                            add_tags_and_relationships(cursor, img_node['id'], img_node.get('tags', []))
                        
                        # Enhanced spatial relationships with better proximity detection
                        for para in current_page_paragraphs:
                            for img_node in current_page_images:
                                if is_spatially_related(para, img_node, proximity_threshold=100):
                                    add_relationship(cursor, para['id'], img_node['id'], 'ILLUSTRATED_BY')
                                    add_relationship(cursor, img_node['id'], para['id'], 'ILLUSTRATES')
                        
                        conn.commit()
                        
                except Exception as e:
                    print(f"⚠️ Database storage failed for page {page_num + 1}: {e}")
                
                pdf_stats['pages'] += 1
            
            # PDF-specific knowledge graph generation
            print(f"🔗 Building knowledge graph for {pdf_name}...")
            try:
                kg_entities = build_knowledge_graph_for_pdf(pdf_path, pdf_paragraphs, pdf_images)
                pdf_stats['kg_entities'] = kg_entities
                print(f"✅ Knowledge graph built with {kg_entities} entities for {pdf_name}")
            except Exception as e:
                print(f"⚠️ Knowledge graph generation failed for {pdf_name}: {e}")
                pdf_stats['kg_entities'] = 0
            
            pdf_processing_stats[pdf_name] = pdf_stats
            print(f"✅ Completed {pdf_name}: {pdf_stats['paragraphs']} paragraphs, {pdf_stats['images']} images, {pdf_stats['kg_entities']} KG entities")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {str(e)}")
            pdf_processing_stats[os.path.basename(pdf_path)] = {'error': str(e)}
        
        finally:
            if 'doc' in locals():
                doc.close()
    
    # Enhanced hybrid knowledge storage building
    print("🧠 Building hybrid knowledge indexes...")
    try:
        hybrid_storage.build_hybrid_indexes(all_paragraphs, all_images)
        
        # Set global variables for backward compatibility
        faiss_index = hybrid_storage.text_faiss_index
        sentence_id_map = hybrid_storage.text_id_map
        
        print("✅ Text FAISS index built with {} paragraphs".format(len(all_paragraphs)))
        print("✅ CLIP index built with {} images".format(len(all_images)))
        
    except Exception as e:
        print(f"❌ Failed to build hybrid indexes: {e}")
        return None
    
    # Cross-PDF relationship analysis
    print("🔍 Analyzing cross-PDF relationships...")
    try:
        cross_pdf_relationships = analyze_cross_pdf_relationships(all_paragraphs, pdf_paths)
        print(f"✅ Found {cross_pdf_relationships} cross-PDF relationships")
    except Exception as e:
        print(f"⚠️ Cross-PDF analysis failed: {e}")
    
    # Final statistics
    total_paragraphs = sum(stats.get('paragraphs', 0) for stats in pdf_processing_stats.values() if 'error' not in stats)
    total_images = sum(stats.get('images', 0) for stats in pdf_processing_stats.values() if 'error' not in stats)
    total_kg_entities = sum(stats.get('kg_entities', 0) for stats in pdf_processing_stats.values() if 'error' not in stats)
    
    print(f"\n✅ Hybrid processing complete!")
    print(f"📊 Total Statistics:")
    print(f"   📚 PDFs processed: {len([s for s in pdf_processing_stats.values() if 'error' not in s])}/{len(pdf_paths)}")
    print(f"   📄 Paragraphs: {total_paragraphs}")
    print(f"   🖼️ Images: {total_images}")
    print(f"   🔗 KG entities: {total_kg_entities}")
    
    # Per-PDF breakdown
    print(f"\n📈 Per-PDF Statistics:")
    for pdf_name, stats in pdf_processing_stats.items():
        if 'error' not in stats:
            print(f"   {pdf_name}: {stats['paragraphs']}P, {stats['images']}I, {stats['kg_entities']}KG")
        else:
            print(f"   {pdf_name}: ❌ {stats['error']}")
    
    return hybrid_storage

def build_knowledge_graph_for_pdf(pdf_path, paragraphs, images):
    """
    Build a knowledge graph specifically for one PDF
    """
    try:
        # Extract entities and relationships from paragraphs
        entities = set()
        relationships = []
        
        for para in paragraphs:
            text = para.get('text', '')
            # Simple entity extraction (can be enhanced with NLP)
            words = text.lower().split()
            
            # Extract key terms (nouns, technical terms)
            key_terms = [word.strip('.,!?;:') for word in words 
                        if len(word) > 3 and word.isalpha()]
            entities.update(key_terms[:10])  # Limit entities per paragraph
        
        # Store knowledge graph in database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Create KG table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    id INTEGER PRIMARY KEY,
                    source_pdf TEXT,
                    entity TEXT,
                    related_entity TEXT,
                    relationship TEXT,
                    strength REAL
                )
            """)
            
            # Insert entities and relationships
            entity_count = 0
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i != j and entity1 != entity2:
                        # Simple co-occurrence relationship
                        cursor.execute("""
                            INSERT OR IGNORE INTO knowledge_graph 
                            (source_pdf, entity, related_entity, relationship, strength)
                            VALUES (?, ?, ?, ?, ?)
                        """, (pdf_path, entity1, entity2, 'co-occurs', 0.5))
                        entity_count += 1
            
            conn.commit()
            
        return len(entities)
        
    except Exception as e:
        print(f"⚠️ KG building error: {e}")
        return 0

def analyze_cross_pdf_relationships(all_paragraphs, pdf_paths):
    """
    Find relationships between content across different PDFs
    """
    try:
        cross_relationships = 0
        pdf_contents = {}
        
        # Group paragraphs by PDF
        for para in all_paragraphs:
            pdf_path = para.get('source_pdf', '')
            if pdf_path not in pdf_contents:
                pdf_contents[pdf_path] = []
            pdf_contents[pdf_path].append(para.get('text', ''))
        
        # Simple cross-PDF similarity analysis
        for pdf1 in pdf_contents:
            for pdf2 in pdf_contents:
                if pdf1 != pdf2:
                    # Find common terms (can be enhanced with semantic similarity)
                    text1 = ' '.join(pdf_contents[pdf1]).lower()
                    text2 = ' '.join(pdf_contents[pdf2]).lower()
                    
                    words1 = set(text1.split())
                    words2 = set(text2.split())
                    
                    common_words = words1.intersection(words2)
                    if len(common_words) > 50:  # Threshold for relationship
                        cross_relationships += 1
        
        return cross_relationships
        
    except Exception as e:
        print(f"⚠️ Cross-PDF analysis error: {e}")
        return 0

def is_spatially_related(para, img_node, proximity_threshold=100):
    """
    Enhanced spatial relationship detection with configurable threshold
    """
    try:
        # Get paragraph and image bounding boxes
        para_bbox = para.get('bbox')
        img_bbox = img_node.get('bbox')
        
        if not para_bbox or not img_bbox:
            return False
        
        # Calculate distance between centers
        para_center = ((para_bbox[0] + para_bbox[2]) / 2, (para_bbox[1] + para_bbox[3]) / 2)
        img_center = ((img_bbox[0] + img_bbox[2]) / 2, (img_bbox[1] + img_bbox[3]) / 2)
        
        distance = ((para_center[0] - img_center[0])**2 + (para_center[1] - img_center[1])**2)**0.5
        
        return distance < proximity_threshold
        
    except Exception as e:
        return False

def get_hybrid_enhanced_answer(query, conversation_history=None, context=None, target_pdf=None):
    """
    Ultimate RAG function optimized for multiple PDFs with all hybrid features integrated
    """
    start_time = time.time()
    global faiss_index, sentence_id_map, knowledge_graph
    
    # Handle special cases (conversation/document summaries)
    if ("conversation" in query.lower() or "chat" in query.lower()) and ("summary" in query.lower()):
        if conversation_history and len(conversation_history) > 0:
            full_history = "\n".join([f"User: {turn[0]}\nAI: {turn[1]}" for turn in conversation_history if turn[1]])
            prompt = f"Please provide a concise summary of the following conversation:\n\n{full_history}"
            return generate_llm_response(prompt), [], ""
        else:
            return "There is no conversation history to summarize yet.", [], ""
    
    if ("document" in query.lower() or "pdf" in query.lower()) and ("summary" in query.lower()):
        # For multi-PDF setup, get summary from all available PDFs
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Get all available PDFs
            available_pdfs = [row[0] for row in cursor.execute("SELECT DISTINCT source_pdf FROM paragraphs").fetchall()]
            
            if not available_pdfs:
                return "No documents have been processed yet.", [], ""
            
            # If target_pdf specified, use it; otherwise use first available
            pdf_to_summarize = target_pdf if target_pdf and target_pdf in available_pdfs else available_pdfs[0]
            
            # Get summary context from the selected PDF
            first_page = "\n".join([row[0] for row in cursor.execute(
                "SELECT text FROM paragraphs WHERE source_pdf = ? ORDER BY page_num ASC LIMIT 3", 
                (pdf_to_summarize,)).fetchall()])
            last_page = "\n".join([row[0] for row in cursor.execute(
                "SELECT text FROM paragraphs WHERE source_pdf = ? ORDER BY page_num DESC LIMIT 2", 
                (pdf_to_summarize,)).fetchall()])
            summary_context = f"Introduction:\n{first_page}\n\nConclusion:\n{last_page}"
        
        prompt = f"Based on the following content from {pdf_to_summarize}, provide a concise summary:\n\n{summary_context}"
        return generate_llm_response(prompt), [], summary_context
    
    # Standard hybrid RAG pipeline for multiple PDFs
    if faiss_index is None:
        return "🚫 Knowledge base not initialized. Please process PDFs first.", [], ""
    
    # STAGE 0: Load knowledge graphs from all available PDFs
    all_knowledge_graphs = {}
    available_pdfs = []
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        available_pdfs = [row[0] for row in cursor.execute("SELECT DISTINCT source_pdf FROM paragraphs").fetchall()]
    
    # Load knowledge graphs for all PDFs
    for pdf_path in available_pdfs:
        try:
            kg = load_graph_from_db(pdf_path)
            if kg is not None:
                all_knowledge_graphs[pdf_path] = kg
                print(f"📊 Loaded knowledge graph for {os.path.basename(pdf_path)}")
            else:
                print(f"⚠️ Could not load knowledge graph for {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"⚠️ Error loading knowledge graph for {os.path.basename(pdf_path)}: {e}")
    
    if not all_knowledge_graphs:
        print("⚠️ No knowledge graphs available, continuing with text-only retrieval")
    else:
        print(f"✅ Loaded {len(all_knowledge_graphs)} knowledge graphs")
    
    # STAGE 1: Enhanced text retrieval with knowledge graph enrichment
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).astype('float32').reshape(1, -1)
    D, I = faiss_index.search(query_embedding, k=15)  # Get more candidates for better coverage
    valid_indices = [i for i, d in zip(I[0], D[0]) if d < 1.5]  # Relaxed threshold for multi-PDF
    
    if not valid_indices:
        print(f"--> [DEBUG] No similar content found for query: '{query}' (tried {len(I[0])} candidates)")
        return "Sorry, I couldn't find relevant content for that question.", [], ""
    
    # STAGE 1.5: Knowledge graph enhancement
    kg_enhanced_results = []
    for i in valid_indices:
        if i in sentence_id_map:
            base_result = {
                'id': sentence_id_map[i]["paragraph_id"],
                'text': sentence_id_map[i]["text"],
                'source_pdf': sentence_id_map[i].get("source_pdf", "unknown"),
                'embedding': query_embedding.flatten(),
                'kg_score': 0.0  # Initialize KG score
            }
            
            # Enhance with knowledge graph if available
            source_pdf = base_result['source_pdf']
            if source_pdf in all_knowledge_graphs:
                kg = all_knowledge_graphs[source_pdf]
                try:
                    # Extract key entities/concepts from the query
                    query_words = set(query.lower().split())
                    kg_boost = 0.0
                    
                    # Check if text contains entities that are highly connected in the knowledge graph
                    text_words = set(base_result['text'].lower().split())
                    common_words = query_words.intersection(text_words)
                    
                    # Simple knowledge graph scoring based on node connections
                    for word in common_words:
                        if word in kg.nodes():
                            # Boost score based on node degree (how connected it is)
                            degree = len(list(kg.neighbors(word)))
                            kg_boost += degree * 0.01  # Small boost per connection
                    
                    base_result['kg_score'] = kg_boost
                    print(f"🔗 KG boost for '{word}': {kg_boost:.3f}") if kg_boost > 0 else None
                    
                except Exception as e:
                    print(f"⚠️ KG processing error for {os.path.basename(source_pdf)}: {e}")
            
            kg_enhanced_results.append(base_result)
    
    # Sort by combined similarity + knowledge graph score
    kg_enhanced_results.sort(key=lambda x: x['kg_score'], reverse=True)
    
    # Apply MMR for diverse text selection on KG-enhanced results
    mmr_ranker = MMRRanker(lambda_param=0.6)  # Slightly favor diversity for multi-PDF
    diverse_results = mmr_ranker.rerank_search_results(query_embedding.flatten(), kg_enhanced_results, k=6)
    paragraph_ids_to_fetch = [r['id'] for r in diverse_results]
    
    # Get context text with source information
    context_paragraphs = []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for para_id in paragraph_ids_to_fetch:
            result = cursor.execute(
                "SELECT text, source_pdf, page_num FROM paragraphs WHERE id = ?", 
                (para_id,)
            ).fetchone()
            if result:
                text, source_pdf, page_num = result
                context_paragraphs.append({
                    'text': text,
                    'source': f"{os.path.basename(source_pdf)} (Page {page_num + 1})"
                })
    
    if not context_paragraphs:
        print(f"--> [DEBUG] No paragraphs retrieved for IDs: {paragraph_ids_to_fetch}")
        return "Sorry, I couldn't find relevant content for that question.", [], ""
    
    # Join context text
    context_text = "\n\n".join([para['text'] for para in context_paragraphs])
    sources_info = "Sources: " + ", ".join(set([para['source'] for para in context_paragraphs]))
    
    print(f"🔍 Retrieved {len(context_paragraphs)} relevant paragraphs from multiple PDFs")
    print(f"📚 {sources_info}")
    print(f"🔗 Knowledge graph enhancement applied to {len([kg for kg in all_knowledge_graphs.values()])} PDFs")
    
    # Generate LLM response
    history_context = ""
    if conversation_history:
        history_to_use = conversation_history[-2:]
        history_text = "\n".join([f"User: {turn[0]}\nAI: {turn[1]}" for turn in history_to_use if turn[1]])
        history_context = f"Recent conversation:\n{history_text}\n\n"
    
    prompt = f"""{history_context}
        Answer only based on Context: {context_text}
        Based on the context above, answer this question: {query}
        If the context does not contain enough information, say "I don't know" or "I apologize, but I cannot answer that question based on the provided context." and DO NOT explain yourself
        If the question is about the document, provide a concise answer.
        Answer:"""
    
    llm_answer = generate_llm_response(prompt)
    
    # STAGE 2: Multi-PDF image retrieval
    clip_retriever = CLIPImageRetriever()
    
    # Get all images from all PDFs (remove target_pdf filter)
    all_images = []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Modified query to get images from all PDFs, not just target_pdf
        img_results = cursor.execute("SELECT * FROM images ORDER BY source_pdf, page_num").fetchall()
        column_names = [description[0] for description in cursor.description]
        all_images = [dict(zip(column_names, row)) for row in img_results]
    
    final_images = []
    
    if all_images:
        try:
            # Build CLIP index if not already built
            if clip_retriever.clip_index is None:
                print("🖼️ Loading CLIP model on cpu...")
                clip_retriever.build_clip_index(all_images)
            
            # Visual search using CLIP across all PDFs
            clip_results = clip_retriever.search_images_by_text(query, k=4)
            
            # Contextual search by page proximity (modified for multi-PDF)
            contextual_results = []
            if paragraph_ids_to_fetch:
                # Get contextual images from pages where relevant text was found
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    for para_id in paragraph_ids_to_fetch[:3]:  # Limit to top 3 paragraphs
                        para_info = cursor.execute(
                            "SELECT source_pdf, page_num FROM paragraphs WHERE id = ?", 
                            (para_id,)
                        ).fetchone()
                        if para_info:
                            source_pdf, page_num = para_info
                            nearby_images = cursor.execute(
                                """SELECT * FROM images 
                                   WHERE source_pdf = ? AND page_num BETWEEN ? AND ?
                                   ORDER BY page_num LIMIT 2""", 
                                (source_pdf, max(0, page_num-1), page_num+1)
                            ).fetchall()
                            contextual_results.extend([dict(zip(column_names, row)) for row in nearby_images])
            
            # Combine and deduplicate results
            all_img_results = clip_results + contextual_results
            seen_img_ids = set()
            
            for img_data in all_img_results:
                if img_data['id'] not in seen_img_ids:
                    final_images.append({
                        'caption': img_data['caption'],
                        'data': img_data['data'],
                        'source': f"{os.path.basename(img_data['source_pdf'])} (Page {img_data['page_num'] + 1})"
                    })
                    seen_img_ids.add(img_data['id'])
                    
                    if len(final_images) >= 3:  # Increased to 3 for better multi-PDF coverage
                        break
        except Exception as e:
            print(f"⚠️ Image retrieval failed: {e}")
            final_images = []
    
    end_time = time.time()
    print(f"⏱️ Hybrid answer generation time: {end_time - start_time:.2f} seconds")
    
    # Fixed return order to match evaluation expectations: (answer, images, context)
    return llm_answer, final_images, context_text

def build_faiss_index(all_paragraphs):
    """Build FAISS index with memory optimization"""
    global embedding_model

    if not all_paragraphs:
        return None, {}

    try:
        # Process in batches to save memory
        batch_size = 32
        embeddings = []

        print(f"📊 Processing {len(all_paragraphs)} paragraphs in batches...")

        for i in range(0, len(all_paragraphs), batch_size):
            batch = all_paragraphs[i:i + batch_size]
            texts = [p['text'] for p in batch]

            batch_embeddings = embedding_model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)

            # Clear memory
            if i % 100 == 0:
                import gc
                gc.collect()

        # Convert to numpy array
        embedding_array = np.array(embeddings, dtype=np.float32)

        # Build FAISS index
        embedding_dim = embedding_array.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embedding_array)

        # Create ID mapping
        id_map = {
            i: {
                "paragraph_id": p["id"],
                "embedding_id": f"embed_{p['id']}",
                "text": p["text"]
            }
            for i, p in enumerate(all_paragraphs)
        }

        print(f"✅ FAISS index built with {index.ntotal} paragraphs")
        return index, id_map

    except Exception as e:
        print(f"❌ FAISS index build failed: {str(e)}")
        return None, {}

def generate_llm_response(prompt):
    """Generate response using GPT4All"""
    global llm_model
    try:
        with llm_model.chat_session():
            response = llm_model.generate(prompt, max_tokens=300, temp=0.7)
            return response
    except Exception as e:
        print(f"❌ LLM generation failed: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now."


def search_by_tags(tag_name, limit=10):
    """Search content by specific tags from the SQLite database."""
    if not os.path.exists(DB_PATH):
        return []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        sql_query = """
            SELECT p.* FROM paragraphs p
            JOIN relationships r ON p.id = r.source_id
            WHERE r.target_id = ? AND r.type = 'HAS_TAG'
            ORDER BY p.page_num
            LIMIT ?
        """
        results = cursor.execute(sql_query, (tag_name.lower(), limit)).fetchall()

        # We need to get column names to reconstruct the dictionary
        column_names = [description[0] for description in cursor.description]

        # Convert list of tuples into list of dictionaries
        final_results = []
        for row in results:
            final_results.append({'paragraph': dict(zip(column_names, row)), 'images': []}) # Images can be added with a second query if needed

        return final_results

def get_available_tags():
    """Gets all available tags and their counts from the SQLite database."""
    if not os.path.exists(DB_PATH):
        return []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # This query counts occurrences of each tag in the relationships table
        sql_query = """
            SELECT target_id, COUNT(source_id) as paragraph_count
            FROM relationships
            WHERE type = 'HAS_TAG'
            GROUP BY target_id
            ORDER BY paragraph_count DESC
            LIMIT 50
        """
        results = cursor.execute(sql_query).fetchall()
        return results # Returns a list of tuples like [('magnet', 5), ('starch', 3)]

def get_document_structure(pdf_path):
    """Gets the hierarchical structure of a document from the SQLite database."""
    if not os.path.exists(DB_PATH):
        return {}
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        sql_query = """
            SELECT page_num, header, section_type
            FROM paragraphs
            WHERE source_pdf = ?
            ORDER BY page_num
        """
        results = cursor.execute(sql_query, (pdf_path,)).fetchall()

        structure = {}
        for row in results:
            page_num, header, section_type = row
            if page_num not in structure:
                structure[page_num] = []
            structure[page_num].append({
                'header': header,
                'section_type': section_type
            })
        return structure

def find_related_images(query):
    """Finds images by searching captions and OCR text in the SQLite database."""
    if not os.path.exists(DB_PATH):
        return []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Use LIKE for a simple text search
        search_term = f"%{query.lower()}%"
        sql_query = """
            SELECT * FROM images
            WHERE LOWER(ocr_text) LIKE ? OR LOWER(caption) LIKE ?
            ORDER BY page_num
            LIMIT 10
        """
        results = cursor.execute(sql_query, (search_term, search_term)).fetchall()

        column_names = [description[0] for description in cursor.description]
        return [dict(zip(column_names, row)) for row in results]

def export_knowledge_base(output_path):
    """Exports the contents of the SQLite database to a JSON file."""
    if not os.path.exists(DB_PATH):
        return None
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        export_data = {
            'paragraphs': [],
            'images': [],
            'metadata': get_statistics() # Assumes get_statistics is already migrated
        }

        # Fetch all paragraphs
        para_rows = cursor.execute("SELECT * FROM paragraphs").fetchall()
        para_cols = [description[0] for description in cursor.description]
        export_data['paragraphs'] = [dict(zip(para_cols, row)) for row in para_rows]

        # Fetch all images, but remove the large base64 data
        img_rows = cursor.execute("SELECT id, caption, ocr_text, page_num, source_pdf, bbox, tags FROM images").fetchall()
        img_cols = [description[0] for description in cursor.description]
        export_data['images'] = [dict(zip(img_cols, row)) for row in img_rows]

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        return export_data

def semantic_search_with_filters(query, tags=None, pdf_source=None, page_range=None):
    """
    Advanced semantic search with filters, fully migrated to SQLite.
    Note: This function is not currently used by the Gradio UI.
    """
    global faiss_index, sentence_id_map

    if faiss_index is None:
        return "Knowledge base not initialized.", []

    # 1. FAISS search to get initial candidates
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).astype('float32').reshape(1, -1)
    D, I = faiss_index.search(query_embedding, k=20)

    # Get paragraph IDs and their scores from FAISS results
    initial_results = {
        sentence_id_map[i]["paragraph_id"]: d
        for i, d in zip(I[0], D[0]) if d < 1.5
    }
    paragraph_ids_to_fetch = list(initial_results.keys())

    if not paragraph_ids_to_fetch:
        return "No relevant content found.", []

    # 2. Build a dynamic SQL query to filter candidates in SQLite
    params = [pdf_source] if pdf_source else []

    # Base query to get paragraphs that are in our candidate list
    sql_query = f"""
        SELECT * FROM paragraphs p
        WHERE p.id IN ({','.join('?' for _ in paragraph_ids_to_fetch)})
    """
    params.extend(paragraph_ids_to_fetch)

    # Add optional filters to the query
    if pdf_source:
        sql_query += " AND p.source_pdf = ?"
    if page_range:
        sql_query += " AND p.page_num BETWEEN ? AND ?"
        params.extend(page_range)
    if tags:
        # This is a more complex filter: find paragraphs that have a relationship to any of the given tags
        tags_placeholder = ','.join('?' for _ in tags)
        sql_query += f"""
            AND p.id IN (
                SELECT r.source_id FROM relationships r
                WHERE r.type = 'HAS_TAG' AND r.target_id IN ({tags_placeholder})
            )
        """
        params.extend(tags)

    # 3. Execute the query and process results
    filtered_results = []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        results = cursor.execute(sql_query, params).fetchall()
        column_names = [description[0] for description in cursor.description]

        for row in results:
            para_dict = dict(zip(column_names, row))
            filtered_results.append({
                'paragraph': para_dict,
                'images': [], # Image fetching can be added as a secondary query
                'similarity_score': initial_results.get(para_dict['id'])
            })

    # Sort by similarity and return top results
    filtered_results.sort(key=lambda x: x['similarity_score'])
    return filtered_results[:5]

# **Speech helper functions**


#!pip install transformers torch librosa pyttsx3 --quiet

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import subprocess # New import for running command-line tools

def transcribe_audio(audio_path):
    # ... (Your existing transcribe_audio function)
    if not audio_path: return ""
    try:
        speech_array, _ = librosa.load(audio_path, sr=16000)
        input_features = whisper_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = whisper_model.generate(input_features)
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""



import torch
import subprocess
import tempfile
import os
from pathlib import Path

def text_to_speech_offline(text, output_path="bot_response_offline.wav", speed=1.25, reference_voice_path="voice_input.wav"):
    """
    Hybrid TTS: Coqui XTTS (primary) with Piper TTS (fallback)
    
    Args:
        text (str): Text to synthesize
        output_path (str): Output file path
        speed (float): Speech speed multiplier (affects Piper only)
        reference_voice_path (str): Path to reference voice for cloning (for Coqui)
    
    Returns:
        str: Path to generated audio file, or None if failed
    """
    if not text:
        print("❌ No text provided for synthesis")
        return None
    
    print(f"🎯 Starting TTS synthesis: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Try Coqui XTTS first (primary)
    coqui_result = _synthesize_with_coqui(text, output_path, reference_voice_path)
    if coqui_result:
        print("✅ Coqui XTTS synthesis successful!")
        return coqui_result
    
    # Fallback to Piper TTS
    print("⚠️ Coqui XTTS failed, falling back to Piper TTS...")
    piper_result = _synthesize_with_piper(text, output_path, speed)
    if piper_result:
        print("✅ Piper TTS fallback synthesis successful!")
        return piper_result
    
    print("❌ Both Coqui XTTS and Piper TTS failed")
    return None

def _synthesize_with_coqui(text, output_path, reference_voice_path):
    """
    Synthesize using Coqui XTTS with voice cloning
    """
    try:
        print("🧠 Attempting Coqui XTTS synthesis...")
        
        # Check if reference voice file exists
        if not os.path.exists(reference_voice_path):
            print(f"⚠️ Reference voice file not found: {reference_voice_path}")
            return None
        
        # Import TTS (only when needed to avoid startup overhead)
        try:
            from TTS.api import TTS
            print("📚 Coqui TTS library imported successfully")
        except ImportError:
            print("❌ Coqui TTS library not installed. Run: pip install TTS")
            return None
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {device}")
        
        # Load XTTS model
        print("🔄 Loading Coqui XTTS v2 model...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("✅ Model loaded successfully")
        
        # Generate speech with voice cloning
        print(f"🗣️ Generating speech with cloned voice from: {reference_voice_path}")
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=reference_voice_path,
            language="en"
        )
        
        # Verify output file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ Coqui XTTS output saved: {output_path} ({os.path.getsize(output_path)} bytes)")
            return output_path
        else:
            print("❌ Coqui XTTS failed - no output file generated")
            return None
            
    except Exception as e:
        print(f"❌ Coqui XTTS error: {str(e)}")
        return None

def _synthesize_with_piper(text, output_path, speed):
    """
    Piper TTS synthesis (fallback method)
    """
    try:
        print("🔄 Using Piper TTS fallback...")
        
        # Piper TTS configuration
        piper_executable = "./piper/build/piper"
        voice_model_path = "./piper_voices/en_US/voice.onnx"
        voice_config_path = "./piper_voices/en_US/voice.json"
        espeak_data_path = "/opt/homebrew/share/espeak-ng-data"
        
        # Verify Piper setup exists
        required_files = [piper_executable, voice_model_path, voice_config_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Piper TTS setup incomplete. Missing files: {missing_files}")
            return None
        
        print("✅ Piper TTS setup verified")
        
        # Create temporary file for Piper output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
            temp_output_path = tmp_audio_file.name
        
        # Build Piper command
        command = [
            piper_executable,
            "--model", voice_model_path,
            "--config", voice_config_path,
            "--output_file", temp_output_path,
            "--length_scale", str(speed),
            "--noise_scale", "0.667",
            "--noise_w", "0.8",
            "--espeak_data", espeak_data_path
        ]
        
        print(f"🔧 Piper command: {' '.join(command[:3])} [...]")
        print(f"📝 Synthesizing text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Run Piper TTS
        result = subprocess.run(
            command, 
            input=text, 
            encoding='utf-8', 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=30  # 30 second timeout
        )
        
        print("✅ Piper synthesis process completed")
        
        # Move temporary file to final output path
        if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
            os.rename(temp_output_path, output_path)
            print(f"✅ Piper output saved: {output_path} ({os.path.getsize(output_path)} bytes)")
            return output_path
        else:
            print("❌ Piper failed - no output file generated")
            return None
        
    except subprocess.TimeoutExpired:
        print("❌ Piper TTS timeout after 30 seconds")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Piper TTS subprocess error:")
        print(f"   Return code: {e.returncode}")
        if e.stderr:
            print(f"   STDERR: {e.stderr}")
        if e.stdout:
            print(f"   STDOUT: {e.stdout}")
        return None
    except Exception as e:
        print(f"❌ Piper TTS unexpected error: {str(e)}")
        return None

# ============================================================================
# USAGE EXAMPLES AND SETUP VERIFICATION
# ============================================================================

def verify_tts_setup():
    """
    Verify that both TTS systems are properly configured
    """
    print("🔍 Verifying TTS setup...")
    
    # Check Coqui TTS
    try:
        import torch
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Coqui TTS available (device: {device})")
    except ImportError:
        print("❌ Coqui TTS not available - install with: pip install TTS")
    
    # Check Piper TTS
    piper_files = [
        "./piper/build/piper",
        "./piper_voices/en_US/voice.onnx", 
        "./piper_voices/en_US/voice.json"
    ]
    
    missing_piper = [f for f in piper_files if not os.path.exists(f)]
    if missing_piper:
        print(f"❌ Piper TTS incomplete - missing: {missing_piper}")
    else:
        print("✅ Piper TTS setup complete")
    
    # Check reference voice
    if os.path.exists("voice_input.wav"):
        size = os.path.getsize("voice_input.wav")
        print(f"✅ Reference voice found: voice_input.wav ({size} bytes)")
    else:
        print("⚠️ Reference voice not found: voice_input.wav")
    
    print("🏁 Setup verification complete")

# Example usage:
"""
# Verify setup first
verify_tts_setup()

# Use the hybrid TTS function
audio_file = text_to_speech_offline(
    text="Hello from Bengaluru! This is a test of the hybrid TTS system.",
    output_path="test_output.wav",
    speed=1.25,
    reference_voice_path="voice_input.wav"
)

if audio_file:
    print(f"🎵 Audio generated successfully: {audio_file}")
else:
    print("❌ TTS synthesis failed")
"""            
# **GRADIO INTERFACE AND MAIN EXECUTION**

# ============================================================================
# SECTION: GRADIO INTERFACE AND MAIN EXECUTION (FINAL VERSION)
# ============================================================================
# ============================================================================
# SECTION: GRADIO INTERFACE AND MAIN EXECUTION (FINAL VERSION)
# ============================================================================

import gradio as gr
from PIL import Image
import base64
from io import BytesIO

# ============================================================================
# SECTION 4: GRADIO INTERFACE AND MAIN EXECUTION
# ============================================================================

def respond(text_in, audio_in, chat_history, selected_pdf):
    """
    Final respond function that yields intermediate updates for better UI responsiveness.
    """
    # --- STAGE 1: Transcribe and give immediate feedback ---
    
    # Determine the user's message from either text or audio
    user_message = text_in or transcribe_audio(audio_in)
    
    if not user_message:
        # If there's no message, do nothing and exit the function
        yield chat_history, None, None, chat_history
        return

    # Immediately add the user's message to the chat history with a "typing" indicator
    chat_history.append([user_message, "_Typing..._"])
    
    # Yield the first update to the UI. This shows the user's question instantly.
    yield chat_history, None, None, chat_history

    # --- STAGE 2: Perform the slow backend tasks ---

    # Get the bot's response (text and images) from your RAG backend
    # We pass the history *without* the last "typing" message for a clean context
    answer_text, images, context_text = get_hybrid_enhanced_answer(user_message, chat_history[:-1], target_pdf=selected_pdf)
    
    # Generate the audio for the bot's response
    audio_filepath = text_to_speech_offline(answer_text)

    # --- STAGE 3: Update the UI with the final answer ---

    # Replace the "typing" indicator with the final text answer
    chat_history[-1][1] = answer_text

    # Prepare the image gallery data
    gallery_data = []
    if images:
        for img_dict in images:
            decoded_bytes = base64.b64decode(img_dict['data'])
            image = Image.open(BytesIO(decoded_bytes))
            gallery_data.append((image, img_dict['caption']))

    # Yield the final, complete update to all UI components
    yield chat_history, audio_filepath, gallery_data, chat_history

def create_enhanced_interface():
    """
    Creates the complete, final Gradio interface with all tabs and functions.
    """
    with gr.Blocks(title="EdgeLearn AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## EdgeLearn AI\nAn AI-powered Smart Tutor chatbot to have a conversation with your PDF documents.")

        pdf_sources_state = gr.State([])

        with gr.Tab("📁 Upload & Process"):
            gr.Markdown("### 1. Upload PDFs for Processing")
            file_input = gr.File(label="Upload PDF files", file_count="multiple", file_types=[".pdf"])
            process_btn = gr.Button("🚀 Process PDFs", variant="primary")
            process_output = gr.Textbox(label="Processing Results", lines=10, max_lines=20)

            def upload_and_process_wrapper(files):
                if not files: return "No files uploaded.", []
                cleanup_database()
                pdf_paths = [file.name for file in files]
                try:
                    process_hybrid_pdfs(pdf_paths)
                    stats = get_statistics()
                    processed_pdfs = stats.get('pdf_sources', [])
                    output_message = (f"✅ Successfully processed {len(pdf_paths)} PDFs.\n" +
                                      f"📊 Statistics: {stats.get('paragraphs', 0)} sections, {stats.get('images', 0)} images, {stats.get('tags', 0)} unique tags.")
                    return output_message, processed_pdfs
                except Exception as e:
                    return f"❌ Error processing PDFs: {str(e)}", []

            process_btn.click(
                upload_and_process_wrapper,
                inputs=[file_input],
                outputs=[process_output, pdf_sources_state]
            )

        with gr.Tab("💬 Chat with your Documents"):
            gr.Markdown("### 2. Have a Conversation")
            clean_history_state = gr.State([])
            pdf_dropdown = gr.Dropdown(label="Select a Document to Chat With", choices=[], interactive=True)
            chatbot_display = gr.Chatbot(label="Conversation", height=400)
            bot_audio_player = gr.Audio(label="Latest Response Audio", autoplay=False)
            gallery_display = gr.Gallery(label="Latest Response Images", show_label=True, columns=4, height="auto")

            with gr.Row():
                user_text_input = gr.Textbox(show_label=False, placeholder="Type your question here...")
            with gr.Row():
                user_audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Or record your question")

            submit_btn = gr.Button("✉️ Send", variant="primary")
            clear_btn = gr.Button("🗑️ Clear Conversation")

            pdf_sources_state.change(
                fn=lambda pdf_list: gr.update(choices=pdf_list),
                inputs=[pdf_sources_state],
                outputs=[pdf_dropdown]
            )

            submit_btn.click(
                respond,
                inputs=[user_text_input, user_audio_input, clean_history_state, pdf_dropdown],
                outputs=[chatbot_display, bot_audio_player, gallery_display, clean_history_state]
            ).then(lambda: (None, None), None, [user_text_input, user_audio_input], queue=False)

            def clear_all():
                return [], None, None, []
            clear_btn.click(clear_all, None, [chatbot_display, bot_audio_player, gallery_display, clean_history_state])

        # (Your other tabs for Search and Statistics go here)

        with gr.Tab("🏷️ Search by Tags"):
            gr.Markdown("## Search Content by Tags")
            with gr.Row():
                with gr.Column(scale=3):
                    tag_input = gr.Textbox(label="Enter tag name", placeholder="e.g., starch, protein, enzyme")
                    tag_search_btn = gr.Button("🔍 Search", variant="primary")
                with gr.Column(scale=1):
                    show_tags_btn = gr.Button("📋 Show Available Tags")
            tag_output = gr.Textbox(label="Search Results", lines=15, max_lines=25)

            def search_by_tag_ui(tag_name):
                if not tag_name.strip(): return "Please enter a tag name."
                results = search_by_tags(tag_name)
                if not results: return f"No content found for tag: '{tag_name}'"
                response = f"Found {len(results)} sections with tag '{tag_name}':\n\n"
                for i, result in enumerate(results[:5], 1):
                    para = result['paragraph']
                    response += f"{i}. {para.get('header', 'Section')} (Page {para['page_num'] + 1})\n   {para['text'][:200]}...\n\n"
                return response

            def show_available_tags_ui():
                tags = get_available_tags()
                if not tags: return "No tags available."
                return "Available tags (top 20):\n\n" + "\n".join([f"• {tag} ({count} sections)" for tag, count in tags[:20]])

            tag_search_btn.click(search_by_tag_ui, inputs=[tag_input], outputs=[tag_output])
            show_tags_btn.click(show_available_tags_ui, outputs=[tag_output])

        with gr.Tab("📊 Statistics"):
            gr.Markdown("## Knowledge Base Statistics")
            stats_btn = gr.Button("📊 Show Statistics", variant="primary")
            stats_output = gr.Textbox(label="Statistics", lines=15, max_lines=25)

            def show_statistics_ui():
                try:
                    stats = get_statistics()
                    response = "📊 Knowledge Base Statistics:\n\n"
                    response += f"📄 Sections: {stats.get('paragraphs', 0)}\n"
                    response += f"🖼️ Images: {stats.get('images', 0)}\n"
                    response += f"🏷️ Unique Tags: {stats.get('tags', 0)}\n"
                    response += f"📚 Total Pages: {stats.get('total_pages', 0)}\n"
                    response += f"📁 PDF Sources: {len(stats.get('pdf_sources', []))}\n\n"
                    if stats.get('pdf_sources'):
                        response += "Sources:\n" + "\n".join([f"• {pdf}" for pdf in stats['pdf_sources']])
                    return response
                except Exception as e:
                    return f"Error retrieving statistics: {str(e)}"
            stats_btn.click(show_statistics_ui, outputs=[stats_output])

    return demo


# ============================================================================
# SECTION 3: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to set up and launch the application.
    """
    global STOPWORDS, embedding_model, llm_model, whisper_processor, whisper_model
    global faiss_index, sentence_id_map

    # --- 1. Download NLTK Data ---
    """print("📥 Downloading necessary NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)"""
    
    # --- 2. Initialize STOPWORDS after download ---
    STOPWORDS = set(stopwords.words('english'))
    print("✅ NLTK data ready.")

    # --- 3. Download and Load AI Models ---
    print("🧠 Loading base models...")
    try:
        # --- THIS IS THE FIX: Download the Piper Voice ---
        voice_dir = "piper_voices/en_US"
        os.makedirs(voice_dir, exist_ok=True)
        onnx_path = os.path.join(voice_dir, "voice.onnx")
        json_path = os.path.join(voice_dir, "voice.json")

        if not os.path.exists(onnx_path) or not os.path.exists(json_path):
            print("🗣️ Downloading Piper voice model...")
            repo_id = "rhasspy/piper-voices"
            onnx_file = "en/en_US/kusal/medium/en_US-kusal-medium.onnx"
            json_file = "en/en_US/kusal/medium/en_US-kusal-medium.onnx.json"
            
            hf_hub_download(repo_id=repo_id, filename=onnx_file, local_dir=voice_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=repo_id, filename=json_file, local_dir=voice_dir, local_dir_use_symlinks=False)
            
            os.rename(os.path.join(voice_dir, onnx_file), onnx_path)
            os.rename(os.path.join(voice_dir, json_file), json_path)
            print("✅ Piper voice downloaded.")
        # -----------------------------------------------

        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        llm_model = GPT4All("Llama-3.2-1B-Instruct-Q4_0.gguf", device=device)
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        print("✅ Successfully loaded base models")
    except Exception as e:
        print(f"❌ Failed to load models: {str(e)}")
        raise

    # --- 4. Initialize Database and FAISS index ---
    if not os.path.exists(DB_PATH):
        initialize_database()
        print(f"✅ New database created at: {DB_PATH}")
    else:
        print("✅ Existing database found.")

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ID_MAP_PATH):
        print("✅ Loading existing knowledge base from files...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(ID_MAP_PATH, 'r') as f:
            sentence_id_map = {int(k): v for k, v in json.load(f).items()}
    else:
        print("⚠️ No existing FAISS index found. Please process PDFs first.")

    # --- 5. Launch the Gradio UI ---
    print("🚀 Starting EdgeLearn AI...")
    demo = create_enhanced_interface()
    demo.launch(
        debug=True,
        share=True,
        #server_name="0.0.0.0",
        show_error=True
    )

def setup_application():
    """
    This function handles all the setup: NLTK, model loading, and database init.
    It does NOT launch the UI.
    """
    # Make sure we can modify the global variables
    global STOPWORDS, embedding_model, llm_model, whisper_processor, whisper_model
    global faiss_index, sentence_id_map

    # --- 1. Download NLTK Data ---
    print("📥 Downloading necessary NLTK data...")
    # A robust check to see if data exists before downloading
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
    
    # --- 2. Initialize STOPWORDS after download ---
    STOPWORDS = set(stopwords.words('english'))
    print("✅ NLTK data ready.")

    # --- 3. Download and Load AI Models ---
    print("🧠 Loading base models...")
    try:
        # --- Download the Piper Voice Model if it doesn't exist ---
        voice_dir = "piper_voices/en_US"
        os.makedirs(voice_dir, exist_ok=True)
        onnx_path = os.path.join(voice_dir, "voice.onnx")
        json_path = os.path.join(voice_dir, "voice.json")

        if not os.path.exists(onnx_path) or not os.path.exists(json_path):
            print("🗣️ Downloading Piper voice model...")
            repo_id = "rhasspy/piper-voices"
            # Using the stable ljspeech voice
            onnx_file = "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx"
            json_file = "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json"
            
            hf_hub_download(repo_id=repo_id, filename=onnx_file, local_dir=voice_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=repo_id, filename=json_file, local_dir=voice_dir, local_dir_use_symlinks=False)
            
            os.rename(os.path.join(voice_dir, onnx_file), onnx_path)
            os.rename(os.path.join(voice_dir, json_file), json_path)
            print("✅ Piper voice downloaded.")
        
        # --- Load the core AI models ---
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ Hardware check: Using '{device}' for compatible models.")
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Using the faster Q2_K model
        llm_model = GPT4All("Llama-3.2-1B-Instruct-Q4_0.gguf", device=device) 
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        
        print("✅ Successfully loaded base models")
    except Exception as e:
        print(f"❌ Failed to load models: {str(e)}")
        raise

    # --- 4. Initialize Database and FAISS index ---
    if not os.path.exists(DB_PATH):
        initialize_database()
        print(f"✅ New database created at: {DB_PATH}")
    else:
        print("✅ Existing database found.")

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ID_MAP_PATH):
        print("✅ Loading existing knowledge base from files...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(ID_MAP_PATH, 'r') as f:
            # Convert string keys from JSON back to integers for FAISS
            sentence_id_map = {int(k): v for k, v in json.load(f).items()}
    else:
        print("⚠️ No existing FAISS index found. Please process PDFs first.")


# ============================================================================
# INCOMPLETE FUNCTIONS THAT NEED COMPLETION
# ============================================================================

def hybrid_upload_wrapper(files):
    """
    INCOMPLETE: Hybrid upload and processing wrapper for Gradio interface
    """
    if not files: 
        return "No files uploaded.", []
    cleanup_database()
    pdf_paths = [file.name for file in files]
    try:
        # Use the new hybrid processing function
        hybrid_storage = process_hybrid_pdfs(pdf_paths)
        stats = get_statistics()
        processed_pdfs = stats.get('pdf_sources', [])
        output_message = (f"✅ Hybrid processing complete for {len(pdf_paths)} PDFs.\n" +
                          f"📊 Enhanced Statistics: {stats.get('paragraphs', 0)} sections, " +
                          f"{stats.get('images', 0)} images, {stats.get('tags', 0)} tags.\n" +
                          f"🔍 CLIP visual search enabled\n" +
                          f"📈 MMR ranking activated\n" +
                          f"🧠 Intelligent chunking applied")
        return output_message, processed_pdfs
    except Exception as e:
        return f"❌ Error in hybrid processing: {str(e)}", []

def hybrid_respond(text_in, audio_in, chat_history, selected_pdf):
    """
    INCOMPLETE: Enhanced respond function using hybrid system
    """
    # Determine the user's message from either text or audio
    user_message = text_in or transcribe_audio(audio_in)
    
    if not user_message:
        yield chat_history, None, None, chat_history
        return

    # Add user message with typing indicator
    chat_history.append([user_message, "_Typing..._"])
    yield chat_history, None, None, chat_history

    # Use the hybrid enhanced answer function
    answer_text,images,context = get_hybrid_enhanced_answer(user_message, chat_history[:-1], target_pdf=selected_pdf)
    
    # Generate audio response
    audio_filepath = text_to_speech_offline(answer_text)

    # Update chat history with final answer
    chat_history[-1][1] = answer_text

    # Prepare image gallery
    gallery_data = []
    if images:
        for img_dict in images:
            try:
                decoded_bytes = base64.b64decode(img_dict['data'])
                image = Image.open(BytesIO(decoded_bytes))
                gallery_data.append((image, img_dict['caption']))
            except Exception as e:
                print(f"⚠️ Error processing image for gallery: {e}")

    yield chat_history, audio_filepath, gallery_data, chat_history

def update_main_function():
    """
    INCOMPLETE: Updated main function to use hybrid system
    """
    global STOPWORDS, embedding_model, llm_model, whisper_processor, whisper_model
    global faiss_index, sentence_id_map

    # Initialize STOPWORDS
    setup_nltk()
    STOPWORDS = set(stopwords.words('english'))
    print("✅ NLTK data ready.")

    # Load models including CLIP
    print("🧠 Loading hybrid models (including CLIP)...")
    try:
        # Download Piper voice if needed
        voice_dir = "piper_voices/en_US"
        os.makedirs(voice_dir, exist_ok=True)
        onnx_path = os.path.join(voice_dir, "voice.onnx")
        json_path = os.path.join(voice_dir, "voice.json")

        if not os.path.exists(onnx_path) or not os.path.exists(json_path):
            print("🗣️ Downloading Piper voice model...")
            repo_id = "rhasspy/piper-voices"
            onnx_file = "en/en_US/kusal/medium/en_US-kusal-medium.onnx"
            json_file = "en/en_US/kusal/medium/en_US-kusal-medium.onnx.json"
            
            hf_hub_download(repo_id=repo_id, filename=onnx_file, local_dir=voice_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=repo_id, filename=json_file, local_dir=voice_dir, local_dir_use_symlinks=False)
            
            os.rename(os.path.join(voice_dir, onnx_file), onnx_path)
            os.rename(os.path.join(voice_dir, json_file), json_path)
            print("✅ Piper voice downloaded.")

        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        llm_model = GPT4All("Llama-3.2-1B-Instruct-Q4_0.gguf", device=device)
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        
        # CLIP will be loaded on-demand by CLIPImageRetriever
        print("✅ Successfully loaded hybrid models")
    except Exception as e:
        print(f"❌ Failed to load models: {str(e)}")
        raise

    # Initialize database
    if not os.path.exists(DB_PATH):
        initialize_database()
        print(f"✅ New database created at: {DB_PATH}")
    else:
        print("✅ Existing database found.")

    # Load existing indexes if available
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ID_MAP_PATH):
        print("✅ Loading existing knowledge base...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(ID_MAP_PATH, 'r') as f:
            sentence_id_map = {int(k): v for k, v in json.load(f).items()}
    else:
        print("⚠️ No existing FAISS index found. Please process PDFs first.")

    # Launch hybrid interface
    print("🚀 Starting EdgeLearn AI Hybrid System...")
    demo = create_hybrid_interface()
    demo.launch(
        debug=True,
        share=True,
        show_error=True
    )

def create_hybrid_interface():
    """
    INCOMPLETE: Complete hybrid interface with all tabs
    """
    with gr.Blocks(title="EdgeLearn AI - Hybrid System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## EdgeLearn AI - Hybrid System\nAdvanced AI tutor with CLIP visual search, MMR ranking, and intelligent chunking.")

        pdf_sources_state = gr.State([])

        with gr.Tab("📁 Upload & Process"):
            gr.Markdown("### 1. Upload PDFs for Hybrid Processing")
            file_input = gr.File(label="Upload PDF files", file_count="multiple", file_types=[".pdf"])
            process_btn = gr.Button("🚀 Process with Hybrid System", variant="primary")
            process_output = gr.Textbox(label="Processing Results", lines=10, max_lines=20)

            process_btn.click(
                hybrid_upload_wrapper,
                inputs=[file_input],
                outputs=[process_output, pdf_sources_state]
            )

        with gr.Tab("💬 Chat with Documents - Enhanced"):
            gr.Markdown("### 2. Hybrid RAG Conversation")
            clean_history_state = gr.State([])
            pdf_dropdown = gr.Dropdown(label="Select Document", choices=[], interactive=True)
            chatbot_display = gr.Chatbot(label="Enhanced Conversation", height=400)
            bot_audio_player = gr.Audio(label="Audio Response", autoplay=False)
            gallery_display = gr.Gallery(label="CLIP-Retrieved Images", show_label=True, columns=4, height="auto")

            with gr.Row():
                user_text_input = gr.Textbox(show_label=False, placeholder="Ask anything...")
            with gr.Row():
                user_audio_input = gr.Audio(sources=["microphone"], type="filepath")

            submit_btn = gr.Button("✉️ Send (Hybrid)", variant="primary")
            clear_btn = gr.Button("🗑️ Clear")

            # Wire up the interface
            pdf_sources_state.change(
                fn=lambda pdf_list: gr.update(choices=pdf_list),
                inputs=[pdf_sources_state],
                outputs=[pdf_dropdown]
            )

            submit_btn.click(
                hybrid_respond,
                inputs=[user_text_input, user_audio_input, clean_history_state, pdf_dropdown],
                outputs=[chatbot_display, bot_audio_player, gallery_display, clean_history_state]
            ).then(lambda: (None, None), None, [user_text_input, user_audio_input], queue=False)

            def clear_all():
                return [], None, None, []
            clear_btn.click(clear_all, None, [chatbot_display, bot_audio_player, gallery_display, clean_history_state])

        # Add other tabs (Search, Statistics) - these would be similar to existing ones
        with gr.Tab("🔍 Advanced Search"):
            gr.Markdown("### Visual + Text Search")
            search_input = gr.Textbox(label="Search Query")
            search_btn = gr.Button("🔍 Hybrid Search")
            search_results = gr.Textbox(label="Results", lines=10)
            
            def hybrid_search(query):
                # This would need implementation
                return f"Hybrid search for: {query}\n(Implementation needed)"
            
            search_btn.click(hybrid_search, inputs=[search_input], outputs=[search_results])

        with gr.Tab("📊 System Status"):
            gr.Markdown("### Hybrid System Statistics")
            status_btn = gr.Button("📊 Show Status")
            status_output = gr.Textbox(label="System Status", lines=15)

            def show_hybrid_status():
                stats = get_statistics()
                status = f"🔧 HYBRID SYSTEM STATUS\n\n"
                status += f"📄 Text Chunks: {stats.get('paragraphs', 0)}\n"
                status += f"🖼️ Images: {stats.get('images', 0)}\n"
                status += f"🏷️ Tags: {stats.get('tags', 0)}\n"
                status += f"📚 PDFs: {len(stats.get('pdf_sources', []))}\n\n"
                status += f"🧠 FAISS Index: {'✅ Active' if faiss_index else '❌ Not loaded'}\n"
                status += f"🎯 CLIP Visual Search: {'✅ Available' if torch.cuda.is_available() else '⚠️ CPU only'}\n"
                status += f"📈 MMR Ranking: ✅ Active\n"
                status += f"🔧 Intelligent Chunking: ✅ Active"
                return status
            
            status_btn.click(show_hybrid_status, outputs=[status_output])

    return demo

# ============================================================================
# ADDITIONAL HELPER FUNCTIONS THAT NEED COMPLETION
# ============================================================================

def save_hybrid_indexes(hybrid_storage, base_path="hybrid_indexes"):
    """
    INCOMPLETE: Save hybrid indexes to disk
    """
    try:
        os.makedirs(base_path, exist_ok=True)
        
        # Save text FAISS index
        if hybrid_storage.text_faiss_index:
            faiss.write_index(hybrid_storage.text_faiss_index, f"{base_path}/text_faiss.idx")
            with open(f"{base_path}/text_id_map.json", 'w') as f:
                json.dump(hybrid_storage.text_id_map, f)
        
        # Save CLIP index if available
        if hybrid_storage.clip_retriever.clip_index:
            faiss.write_index(hybrid_storage.clip_retriever.clip_index, f"{base_path}/clip_faiss.idx")
            with open(f"{base_path}/clip_metadata.json", 'w') as f:
                json.dump(hybrid_storage.clip_retriever.image_metadata, f)
        
        print("✅ Hybrid indexes saved successfully")
        return True
    except Exception as e:
        print(f"❌ Error saving hybrid indexes: {e}")
        return False

def load_hybrid_indexes(base_path="hybrid_indexes"):
    """
    INCOMPLETE: Load hybrid indexes from disk
    """
    try:
        hybrid_storage = HybridKnowledgeStorage()
        
        # Load text index
        text_idx_path = f"{base_path}/text_faiss.idx"
        text_map_path = f"{base_path}/text_id_map.json"
        
        if os.path.exists(text_idx_path) and os.path.exists(text_map_path):
            hybrid_storage.text_faiss_index = faiss.read_index(text_idx_path)
            with open(text_map_path, 'r') as f:
                hybrid_storage.text_id_map = {int(k): v for k, v in json.load(f).items()}
        
        # Load CLIP index
        clip_idx_path = f"{base_path}/clip_faiss.idx"
        clip_meta_path = f"{base_path}/clip_metadata.json"
        
        if os.path.exists(clip_idx_path) and os.path.exists(clip_meta_path):
            hybrid_storage.clip_retriever.clip_index = faiss.read_index(clip_idx_path)
            with open(clip_meta_path, 'r') as f:
                hybrid_storage.clip_retriever.image_metadata = {int(k): v for k, v in json.load(f).items()}
        
        print("✅ Hybrid indexes loaded successfully")
        return hybrid_storage
    except Exception as e:
        print(f"❌ Error loading hybrid indexes: {e}")
        return None

def launch_app():
    """This function only launches the Gradio UI."""
    print("🚀 Starting EdgeLearn AI...")
    demo = create_hybrid_interface()
    demo.launch()

if __name__ == "__main__":
    # When running app.py directly, do all setup, then launch the app.
    setup_application()
    launch_app()

"""
# @title ▶️ Standalone Test for Piper TTS Voice Customization
import time
import subprocess
import os
import tempfile
from IPython.display import Audio, display

def text_to_speech_offline(text, output_path="bot_response.wav", length_scale=1.0, noise_scale=0.5, noise_w=0.5):

    start_time=time.time()
    if not text:
        return None

    # Define paths
    piper_executable = "/content/piper/piper"
    voice_model_path = "/content/piper_voices/en_US/voice.onnx"
    voice_config_path = "/content/piper_voices/en_US/voice.json"

    # Set environment variable for helper libraries
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = "/content/piper:" + my_env.get("LD_LIBRARY_PATH", "")

    # Build the command with all customization arguments
    command = [
        piper_executable,
        "--model", voice_model_path,
        "--config", voice_config_path,
        "--output_file", output_path,
        "--length_scale", str(length_scale),
        "--noise_scale", str(noise_scale),
        "--noise_w", str(noise_w)
    ]

    try:
        print(f"\nSynthesizing (Speed: {length_scale}): '{text}'")
        # Piper reads text from standard input, so we pass it via the 'input' argument
        subprocess.run(command, input=text, encoding='utf-8', check=True, capture_output=True, env=my_env)
        print(f"✅ Synthesis complete. Audio saved to {output_path}")
        print(f"Time taken: {time.time()-start_time}")
        return output_path
    except Exception as e:
        print(f"❌ An error occurred during Piper TTS synthesis: {e}")
        return None

--- TEST EXECUTION ---
test_sentence = " To test for starch in food, you can use iodine solution and observe if the color changes on the food item. Here's a step-by-step guide:
Place a small piece of each food item (e.g., bread, rice, or pasta) on a separate dish.
Dip 2-3 drops of diluted iodine solution into a dropper.
Put the drops onto each food item and observe if there are any changes in color.
If you notice that:
The blue-black colour indicates the presence of starch (starch contains iodine, which turns black when exposed to it).
No change or only slight discoloration occurs on the food items.
This is a simple test for detecting starch."

# 1. Generate audio at normal speed
normal_audio_path = text_to_speech_offline(test_sentence, output_path="output_normal.wav", length_scale=1.0)

# 2. Generate audio at a slower speed
slow_audio_path = text_to_speech_offline(test_sentence, output_path="output_slow.wav", length_scale=1.2)

# 3. Generate audio at a very slow speed
very_slow_audio_path = text_to_speech_offline(test_sentence, output_path="output_very_slow.wav", length_scale=1.5)

# --- Display the results ---
if normal_audio_path:

    print("\n--- Normal Speed (1.0) ---")
    display(Audio(normal_audio_path))

if slow_audio_path:
    print("\n--- Slower Speed (1.2) ---")
    display(Audio(slow_audio_path))

if very_slow_audio_path:
    print("\n--- Very Slow Speed (1.5) ---")
    display(Audio(very_slow_audio_path))

# prompt: download file generated in the above cell

from google.colab import files
files.download('output_very_slow.wav')


# @title ▶️ Standalone Test to Analyze Audio Quality (SNR)

# First, ensure necessary libraries are installed
!pip install numpy scipy librosa -q

import numpy as np
import librosa
from scipy.io import wavfile
from IPython.display import display, Markdown

def calculate_snr(audio_path):

    try:
        # Load the audio file
        amplitude, sample_rate = librosa.load(audio_path, sr=None)

        # Calculate the power of the signal
        signal_power = np.sum(amplitude ** 2) / len(amplitude)

        # A simple way to estimate noise is to find quiet parts of the audio.
        # We'll consider segments with energy in the bottom 20% to be noise.
        frame_length = 2048
        hop_length = 512
        energy = np.array([
            sum(abs(amplitude[i:i+frame_length]**2))
            for i in range(0, len(amplitude), hop_length)
        ])

        # Find the threshold for quiet parts
        energy_threshold = np.percentile(energy, 20)

        # Extract noise segments
        noise_amplitude = amplitude[np.repeat(energy < energy_threshold, hop_length)[:len(amplitude)]]

        if len(noise_amplitude) == 0:
            # If no noise is detected (very clean signal), we can't calculate SNR this way.
            return float('inf') # Represents an infinitely clean signal

        # Calculate the power of the noise
        noise_power = np.sum(noise_amplitude ** 2) / len(noise_amplitude)

        if noise_power == 0:
            return float('inf') # Avoid division by zero for perfect silence

        # SNR in decibels (dB)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    except Exception as e:
        print(f"Error calculating SNR: {e}")
        return None

# --- EXAMPLE USAGE ---
# Assume you have your generated audio file from Piper
# For this example, let's use the 'output_slow_clear.wav' we made earlier.
# If it doesn't exist, you'll need to generate an audio file first.
generated_audio_file = "output_very_slow.wav"

if os.path.exists(generated_audio_file):
    snr_value = calculate_snr(generated_audio_file)

    if snr_value is not None:
        display(Markdown(f"### 🔊 Audio Quality Analysis for `{generated_audio_file}`"))
        display(Markdown(f"**Signal-to-Noise Ratio (SNR): {snr_value:.2f} dB**"))

        # Interpretation
        interpretation = ""
        if snr_value > 20:
            interpretation = "Excellent quality. The speech is very clear compared to any background noise."
        elif 15 <= snr_value <= 20:
            interpretation = "Good quality. The speech is clearly audible with minimal noise."
        elif 10 <= snr_value < 15:
            interpretation = "Fair quality. Some noise may be perceptible, but speech is understandable."
        else:
            interpretation = "Poor quality. Background noise is significant and may interfere with speech clarity."

        display(Markdown(f"**Interpretation:** {interpretation}"))
else:
    print(f"File not found: '{generated_audio_file}'. Please generate an audio file first to analyze it.")"""