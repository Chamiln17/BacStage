
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List
from tqdm import tqdm

class TextEmbeddingExtractor:
    """
    Extracts semantic embeddings using BERT-based models (default: AraBERT v2).
    """
    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading {model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with memory optimization for CPU
            if self.device == "cpu":
                print("   💡 Using low memory mode for CPU...")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True  # Reduces memory footprint during loading
                )
            else:
                self.model = AutoModel.from_pretrained(model_name)
            
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print("   ✅ Model loaded successfully!")
            
        except OSError as e:
            if "paging file" in str(e).lower() or "1455" in str(e):
                print("\n❌ ERROR: Not enough virtual memory to load the model")
                print("\n📋 Quick Fix Options:")
                print("1. Increase Windows paging file size (see docs/02_model_baseline_testing/fix_memory_error.md)")
                print("2. Close other applications to free RAM")
                print("3. Restart computer and try again")
                print("4. Enable GPU if available (10x faster anyway!)")
                raise RuntimeError(
                    "Insufficient virtual memory. Please increase Windows paging file size or enable GPU. "
                    "See docs/02_model_baseline_testing/fix_memory_error.md for detailed instructions."
                ) from e
            else:
                raise

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed.
            batch_size: Number of texts to process at once.
            
        Returns:
            numpy.ndarray: Array of shape (n_texts, 768) containing embeddings.
        """
        embeddings = []
        
        # Ensure texts are strings (handle NaNs or non-strings if passed)
        clean_texts = [str(t) if t is not None else "" for t in texts]
        
        for i in tqdm(range(0, len(clean_texts), batch_size), desc="Extracting embeddings"):
            batch = clean_texts[i : i + batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Use [CLS] token embedding (first token)
            # Shape: (batch_size, 768)
            batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
            
        if not embeddings:
            return np.empty((0, 768))
            
        return np.vstack(embeddings)
