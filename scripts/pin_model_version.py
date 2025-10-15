#!/usr/bin/env python3
"""
Pin exact HuggingFace model revision for reproducibility
"""

import os
import sys

sys.path.insert(0, "src")

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def pin_model_version(model_name: str = "intfloat/multilingual-e5-large"):
    """Get and display exact model revision with GPT improvements."""

    logger.info(f"📌 Pinning model: {model_name}")

    # Load model
    model = SentenceTransformer(model_name)

    # Get revision from model config (GPT improvement: more comprehensive)
    revision = None
    try:
        # Try multiple approaches to get revision
        if hasattr(model, '_model_card_data') and model._model_card_data:
            revision = getattr(model._model_card_data, 'model_id', None)

        # Fallback: check model config
        if not revision and hasattr(model, '_model_config'):
            revision = model._model_config.get('_commit_hash', None)

        # Fallback: get from transformers model
        if not revision and hasattr(model, '_modules') and len(model._modules) > 0:
            first_module = model._modules['0']
            if hasattr(first_module, 'auto_model') and hasattr(first_module.auto_model, 'config'):
                config = first_module.auto_model.config
                revision = getattr(config, '_commit_hash', None)

        # Fallback: try to get from cache
        if not revision:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            logger.info(f"   Cache directory: {cache_dir}")
            # Note: Could scan cache for model-specific revision info

    except Exception as e:
        logger.warning(f"⚠️ Error getting model revision: {e}")

    # Display results
    if revision:
        logger.info(f"✅ Model revision: {revision}")
        logger.info("\n📝 To pin in requirements.txt, use:")
        logger.info("   # Pinned model revision for reproducibility")
        logger.info(f"   # {model_name}@{revision}")
        logger.info("\n📝 To use in code:")
        logger.info(f"   SentenceTransformer('{model_name}', revision='{revision}')")
    else:
        logger.warning("⚠️ Could not determine model revision")
        logger.info("   Using latest version (not recommended for production)")

    # Display model info
    logger.info("\n📊 Model Info:")
    logger.info(f"   Name: {model_name}")
    logger.info(f"   Max seq length: {model.max_seq_length}")

    # GPT improvement: handle dimension properly
    try:
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info(f"   Embedding dim: {embedding_dim}")
    except Exception as e:
        logger.warning(f"   Embedding dim: Could not determine ({e})")

    # Log current encoder metadata for tracking
    logger.info("\n📋 Encoder Metadata for Tracking:")
    logger.info(f"   model_name: '{model_name}'")
    logger.info(f"   model_revision: '{revision or 'unknown'}'")

    return revision


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    revision = pin_model_version()

    if revision:
        print(f"\n🎯 SUCCESS: Model revision determined: {revision}")
        sys.exit(0)
    else:
        print("\n⚠️ WARNING: Could not determine model revision")
        sys.exit(1)
