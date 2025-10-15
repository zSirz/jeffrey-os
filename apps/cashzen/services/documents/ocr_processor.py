"""
Processeur OCR pour l'extraction de texte des documents
Prend en charge différents formats de documents (PDF, images)
"""

import logging
import mimetypes
import os
import tempfile

import pytesseract
import requests
from dotenv import load_dotenv
from PIL import Image

# Configuration du logging
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Clé API pour OCR.space (si disponible)
OCR_API_KEY = os.getenv("OCR_API_KEY", "")


class OCRProcessor:
    """
    Classe pour l'extraction de texte des documents
    """

    def __init__(self):
        """
        Initialise le processeur OCR
        """
        # Configuration de pytesseract sur différentes plateformes
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # Sur macOS et Linux, pytesseract utilise l'installation par défaut

    def extract_text(self, file_path: str) -> str:
        """
        Extrait le texte d'un document

        Args:
            file_path: Chemin vers le fichier

        Returns:
            str: Texte extrait
        """
        try:
            # Déterminer le type de fichier
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                return self._extract_text_from_image(file_path)
            else:
                logger.error(f"Format de fichier non pris en charge: {file_ext}")
                return ""

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du texte: {str(e)}")
            return ""

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extrait le texte d'un fichier PDF

        Args:
            file_path: Chemin vers le fichier PDF

        Returns:
            str: Texte extrait
        """
        try:
            # Essayer d'abord d'extraire le texte directement du PDF
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(file_path)

                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"

                # Si le texte est significatif, le retourner
                if len(text.strip()) > 100:
                    return text

                # Sinon, essayer avec l'OCR
                logger.info("Extraction directe insuffisante, recours à l'OCR")

            except Exception as e:
                logger.warning(f"Extraction directe échouée: {str(e)}")

            # Convertir les pages PDF en images et appliquer l'OCR
            from pdf2image import convert_from_path

            all_text = ""

            # Convertir les pages en images
            images = convert_from_path(file_path)

            # Pour chaque image, extraire le texte
            for i, image in enumerate(images):
                # Sauvegarder l'image temporairement
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                    temp_path = temp.name
                    image.save(temp_path, 'JPEG')

                # Extraire le texte
                page_text = self._extract_text_from_image(temp_path)
                all_text += f"--- Page {i + 1} ---\n{page_text}\n\n"

                # Supprimer le fichier temporaire
                os.unlink(temp_path)

            return all_text

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du texte du PDF: {str(e)}")
            return ""

    def _extract_text_from_image(self, file_path: str) -> str:
        """
        Extrait le texte d'une image

        Args:
            file_path: Chemin vers le fichier image

        Returns:
            str: Texte extrait
        """
        try:
            # Essayer d'abord avec pytesseract local
            try:
                # Prétraiter l'image pour améliorer la qualité de l'OCR
                image = self._preprocess_image(file_path)

                # Configurer pytesseract pour les langues françaises et anglaises
                custom_config = r'--oem 3 --psm 6'

                # Extraire le texte
                try:
                    # Essayer avec les langues françaises et anglaises
                    text = pytesseract.image_to_string(image, config=custom_config + ' -l fra+eng')
                except Exception as lang_error:
                    # Si les langues ne sont pas disponibles, utiliser la configuration par défaut
                    logger.warning(
                        f"Erreur avec les langues spécifiées: {lang_error}. Utilisation de la configuration par défaut."
                    )
                    text = pytesseract.image_to_string(image, config=custom_config)

                # Si le texte est significatif, le retourner
                if len(text.strip()) > 10:
                    return text

                # Sinon, essayer avec l'API OCR
                logger.info("OCR local insuffisant, recours à l'API OCR")

            except Exception as e:
                logger.warning(f"OCR local échoué: {str(e)}")

            # Si disponible, utiliser l'API OCR.space
            if OCR_API_KEY:
                return self._extract_text_with_api(file_path)

            return ""

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du texte de l'image: {str(e)}")
            return ""

    def _preprocess_image(self, file_path: str) -> Image.Image:
        """
        Prétraite une image pour améliorer la qualité de l'OCR

        Args:
            file_path: Chemin vers le fichier image

        Returns:
            Image.Image: Image prétraitée
        """
        # Ouvrir l'image
        image = Image.open(file_path)

        # Convertir en mode RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Redimensionner si l'image est trop grande
        max_size = 2000
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # Augmenter le contraste
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Augmenter la netteté
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)

        return image

    def _extract_text_with_api(self, file_path: str) -> str:
        """
        Extrait le texte d'une image avec l'API OCR.space

        Args:
            file_path: Chemin vers le fichier image

        Returns:
            str: Texte extrait
        """
        try:
            url = 'https://api.ocr.space/parse/image'

            # Préparer le fichier
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            # Déterminer le type MIME
            mime_type = mimetypes.guess_type(file_path)[0]
            if not mime_type:
                mime_type = 'application/octet-stream'

            # Préparer les données
            payload = {
                'apikey': OCR_API_KEY,
                'language': 'fre',  # French
                'isOverlayRequired': False,
                'detectOrientation': True,
                'scale': True,
                'OCREngine': 2,  # Engine 2 is better for receipts and documents
            }

            files = {'file': (os.path.basename(file_path), file_bytes, mime_type)}

            # Faire la requête
            response = requests.post(url, files=files, data=payload)

            if response.status_code != 200:
                logger.error(f"Erreur API OCR: {response.status_code} - {response.text}")
                return ""

            # Parser la réponse
            result = response.json()

            if not result.get('IsErroredOnProcessing'):
                parsed_results = result.get('ParsedResults', [])
                if parsed_results:
                    return parsed_results[0].get('ParsedText', '')

            logger.error(f"Erreur lors du traitement OCR: {result.get('ErrorMessage', 'Erreur inconnue')}")
            return ""

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du texte avec l'API: {str(e)}")
            return ""
