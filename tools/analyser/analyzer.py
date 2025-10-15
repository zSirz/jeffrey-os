"""Analyseur intelligent avec GPU acceleration et NLP."""

import ast
import hashlib
import re
from pathlib import Path

# En haut du fichier
try:
    import spacy

    # Tenter GPU explicitement
    try:
        spacy.require_gpu()
        NLP_AVAILABLE = True
        GPU_ENABLED = True
        print("✅ GPU activé pour NLP")
    except:
        NLP_AVAILABLE = True
        GPU_ENABLED = False
        print("ℹ️ NLP en mode CPU")

    # Charger modèle multilingue si disponible
    try:
        nlp = spacy.load("xx_ent_wiki_sm", disable=['parser', 'ner'])
    except:
        try:
            nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        except:
            nlp = None
            NLP_AVAILABLE = False
except ImportError:
    NLP_AVAILABLE = False
    GPU_ENABLED = False
    nlp = None

# Hash ultra-rapide BLAKE3
try:
    from blake3 import blake3

    HASH_FUNC = blake3
except:
    HASH_FUNC = hashlib.sha256


class IntelligentAnalyzer:
    """Analyseur avec extraction de concepts et métriques avancées."""

    # Patterns de détection optimisés
    EMOTIONAL_PATTERNS = re.compile(
        r'\b(emotion|feeling|mood|sentiment|happy|sad|angry|fear|joy|love|hate)\b', re.IGNORECASE
    )

    AI_PATTERNS = re.compile(
        r'\b(neural|network|model|train|predict|classify|embedding|transformer|'
        r'attention|lstm|gru|cnn|rnn|bert|gpt)\b',
        re.IGNORECASE,
    )

    SECURITY_PATTERNS = re.compile(
        r'\b(encrypt|decrypt|auth|token|password|secret|key|certificate|ssl|tls)\b', re.IGNORECASE
    )

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and NLP_AVAILABLE
        self.cache = {}

    def hash_file(self, filepath: Path, partial: bool = False) -> str:
        """Hash BLAKE3 ultra-rapide (ou SHA256 fallback)."""
        hasher = HASH_FUNC()

        try:
            with open(filepath, 'rb') as f:
                if partial:
                    # Hash partiel (128KB) pour déduplication rapide
                    data = f.read(128 * 1024)
                    hasher.update(data)
                    return f"P:{hasher.hexdigest()[:32]}"
                else:
                    # Hash complet par chunks
                    while chunk := f.read(1024 * 1024):  # 1MB chunks
                        hasher.update(chunk)
                    return f"F:{hasher.hexdigest()}"
        except:
            return "ERROR"

    def analyze_python_fast(self, filepath: Path) -> dict:
        """Analyse AST Python optimisée."""
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')

            # Parse AST
            tree = ast.parse(content)

            # Métriques rapides
            metrics = {
                'lines': content.count('\n') + 1,
                'classes': [],
                'functions': [],
                'imports': [],
                'complexity': 0,
                'has_tests': False,
                'has_docs': False,
            }

            # Un seul parcours de l'AST pour tout extraire
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    metrics['classes'].append(node.name)
                    if ast.get_docstring(node):
                        metrics['has_docs'] = True

                elif isinstance(node, ast.FunctionDef):
                    metrics['functions'].append(node.name)
                    if node.name.startswith('test_'):
                        metrics['has_tests'] = True
                    if ast.get_docstring(node):
                        metrics['has_docs'] = True

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        metrics['imports'].append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        metrics['imports'].append(node.module)

                # Complexité cyclomatique simplifiée
                elif isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    metrics['complexity'] += 1

            # Extraction de concepts
            metrics['concepts'] = self.extract_concepts(content)

            return metrics

        except Exception as e:
            return {'error': str(e)}

    def extract_concepts(self, content: str) -> dict[str, list[str]]:
        """Extraction rapide de concepts."""
        concepts = {'emotional': [], 'ai': [], 'security': []}

        # Limiter la taille pour performance
        sample = content[:50000] if len(content) > 50000 else content

        # Extraction par regex (rapide)
        if matches := self.EMOTIONAL_PATTERNS.findall(sample):
            concepts['emotional'] = list(set(matches[:10]))

        if matches := self.AI_PATTERNS.findall(sample):
            concepts['ai'] = list(set(matches[:10]))

        if matches := self.SECURITY_PATTERNS.findall(sample):
            concepts['security'] = list(set(matches[:10]))

        # NLP si disponible et GPU
        if self.use_gpu and len(sample) < 10000:
            try:
                doc = nlp(sample[:5000])
                # Extraction d'entités pertinentes
                entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG', 'TECH']]
                if entities:
                    concepts['entities'] = entities[:10]
            except:
                pass

        return concepts
