
import os
import sys

import numpy as np

from src.features.text_embeddings import TextEmbeddingExtractor

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_arabert_extractor():
    print("Testing TextEmbeddingExtractor with AraBERT...")
    try:
        extractor = TextEmbeddingExtractor() # Defaults to AraBERT

        # Test with educational/MSA text (video titles)
        texts = [
            "ملخص شامل في الدوال الأسية باك 2024",
            "حل تمرين رائع في الأعداد المركبة",
            "مراجعة نهائية في الفيزياء النووية",
            "نصائح للتفوق في البكالوريا",
            "أهم المقالات الفلسفية المقترحة"
        ]

        embeddings = extractor.get_embeddings(texts)

        print(f"Input texts: {len(texts)}")
        print(f"Output shape: {embeddings.shape}")

        # Verify shape (AraBERT base is also 768 dim)
        assert embeddings.shape == (5, 768), f"Expected (5, 768), got {embeddings.shape}"

        # Verify diversity
        assert not np.allclose(embeddings[0], embeddings[1]), "Embeddings should be different"
        assert not np.all(embeddings == 0), "Embeddings should not be all zeros"

        print("✅ Test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise e

if __name__ == "__main__":
    test_arabert_extractor()
