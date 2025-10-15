"""
Service de classification automatique de documents pour CashZen
Utilise OpenAI pour analyser et catégoriser automatiquement les documents
"""

import hashlib
import json
import logging
import re
import traceback
from datetime import datetime
from typing import Any

from services.openai_service import get_openai_service

# Configuration du logging
logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Service de classification et d'analyse de documents
    """

    def __init__(self):
        """Initialisation du service de classification de documents"""
        self.openai_service = get_openai_service()

        # Catégories principales de documents
        self.categories = {
            "banque": ["relevé bancaire", "extrait de compte", "avis d'opération"],
            "facture": ["facture", "quittance", "reçu", "note de frais"],
            "impot": ["impôt", "taxe", "déclaration fiscale", "avis d'imposition"],
            "assurance": ["assurance", "police", "cotisation", "sinistre"],
            "sante": ["santé", "médical", "remboursement", "ordonnance"],
            "identite": ["identité", "passeport", "carte", "permis"],
            "logement": ["bail", "loyer", "propriété", "habitation"],
            "transport": ["transport", "véhicule", "carte grise", "assurance auto"],
            "contrat": ["contrat", "convention", "accord", "conditions générales"],
            "autre": [],
        }

        # Mots-clés importants à identifier
        self.mots_cles_importants = [
            "urgent",
            "échéance",
            "délai",
            "paiement",
            "montant",
            "total",
            "date limite",
            "référence",
            "client",
            "numéro",
            "compte",
        ]

        # Conservation des mots-clés de l'ancienne version
        self.keywords = {
            "facture": [
                "facture",
                "invoice",
                "récapitulatif",
                "achat",
                "achats",
                "total",
                "montant",
                "tva",
                "ht",
                "ttc",
                "client",
                "paiement",
                "échéance",
                "règlement",
                "commande",
                "livraison",
                "article",
                "articles",
                "quantité",
                "prix unitaire",
                "prix total",
                "remise",
                "avoir",
                "bon de livraison",
                "payé",
                "à payer",
                "acompte",
                "solde",
                "débit",
                "crédit",
            ],
            "contrat": [
                "contrat",
                "convention",
                "accord",
                "conditions générales",
                "engagement",
                "signataire",
                "soussigné",
                "parties",
                "obligations",
                "durée",
                "terme",
                "clause",
                "clauses",
                "résilier",
                "résiliation",
                "préavis",
                "avenant",
                "accepte",
                "acceptation",
                "signature",
                "signé",
                "contractuel",
                "contractuelle",
            ],
            "releve_bancaire": [
                "relevé",
                "compte",
                "bancaire",
                "banque",
                "opérations",
                "solde",
                "dépôt",
                "retrait",
                "virement",
                "prélèvement",
                "débit",
                "crédit",
                "carte",
                "chèque",
                "agence",
                "iban",
                "bic",
                "swift",
                "sepa",
                "intérêts",
                "frais bancaires",
                "commission",
                "découvert",
                "autorisation",
                "devise",
                "date valeur",
                "date opération",
                "rib",
                "titulaire",
            ],
            "impot": [
                "impôt",
                "impôts",
                "fiscal",
                "fiscale",
                "taxe",
                "taxes",
                "imposition",
                "revenu",
                "revenus",
                "déclaration",
                "imposable",
                "déduction",
                "déductible",
                "contribuable",
                "taux",
                "barème",
                "tranche",
                "administration fiscale",
                "trésor public",
                "avis",
                "cotisation",
                "ifi",
                "ir",
                "is",
                "csg",
                "crds",
                "prélèvement",
                "à la source",
                "réduction",
                "crédit d'impôt",
                "avoir fiscal",
            ],
            "assurance": [
                "assurance",
                "assuré",
                "assureur",
                "police",
                "contrat",
                "garantie",
                "garanties",
                "sinistre",
                "indemnité",
                "indemnisation",
                "prime",
                "cotisation",
                "franchise",
                "échéance",
                "souscripteur",
                "bénéficiaire",
                "couverture",
                "risque",
                "risques",
                "responsabilité",
                "dommage",
                "dommages",
                "accident",
                "vie",
                "décès",
                "invalidité",
                "incapacité",
                "maladie",
                "santé",
                "habitation",
                "véhicule",
                "auto",
                "multirisque",
            ],
            "sante": [
                "médecin",
                "médical",
                "médicale",
                "ordonnance",
                "prescription",
                "traitement",
                "consultation",
                "patient",
                "hôpital",
                "clinique",
                "pharmacie",
                "médicament",
                "médicaments",
                "posologie",
                "sécurité sociale",
                "remboursement",
                "mutuelle",
                "complémentaire",
                "santé",
                "maladie",
                "pathologie",
                "symptôme",
                "diagnostic",
                "thérapie",
                "analyses",
                "examen",
                "radiologie",
                "scanner",
                "irm",
                "laboratoire",
                "chirurgie",
            ],
            "identite": [
                "passeport",
                "carte nationale d'identité",
                "carte d'identité",
                "permis de conduire",
                "état civil",
                "acte de naissance",
                "livret de famille",
                "nationalité",
                "ressortissant",
                "citoyenneté",
                "titre de séjour",
                "visa",
                "préfecture",
                "ambassade",
                "consulat",
                "extrait",
                "mairie",
                "attestation",
                "justificatif",
                "identité",
                "nom",
                "prénom",
                "date de naissance",
                "lieu de naissance",
                "sexe",
                "taille",
                "domicile",
            ],
            "transport": [
                "billet",
                "ticket",
                "voyage",
                "trajet",
                "transport",
                "itinéraire",
                "horaire",
                "train",
                "avion",
                "bus",
                "métro",
                "tram",
                "tramway",
                "navette",
                "passager",
                "réservation",
                "place",
                "siège",
                "classe",
                "aller",
                "retour",
                "aller-retour",
                "escale",
                "correspondance",
                "gare",
                "aéroport",
                "station",
                "terminal",
                "vol",
                "compagnie",
                "air france",
                "sncf",
                "tgv",
            ],
            "logement": [
                "bail",
                "location",
                "loyer",
                "propriétaire",
                "locataire",
                "habitation",
                "résidence",
                "logement",
                "appartement",
                "maison",
                "immeuble",
                "pièce",
                "chambre",
                "cuisine",
                "salon",
                "salle de bain",
                "sanitaire",
                "meublé",
                "non meublé",
                "état des lieux",
                "caution",
                "dépôt de garantie",
                "préavis",
                "congé",
                "charges",
                "copropriété",
                "syndic",
                "diagnostics",
                "dpe",
                "surface",
                "adresse",
                "domicile",
            ],
        }

    def classify_document(self, texte_document: str, nom_fichier: str = "") -> dict[str, Any]:
        """
        Classifie un document en fonction de son contenu

        Args:
            texte_document: Texte extrait du document
            nom_fichier: Nom du fichier (optionnel)

        Returns:
            Dict: Résultat de la classification
        """
        try:
            # Si le texte est très long, le tronquer pour l'API
            texte_tronque = texte_document[:7000] if len(texte_document) > 7000 else texte_document

            # Tenter d'utiliser l'IA pour analyser le document
            try:
                resultats_analyse = self.openai_service.document_analyzer.analyser_document(texte_tronque)

                # Vérifier si l'analyse IA a réussi
                if "erreur" not in resultats_analyse:
                    # Enrichir les résultats
                    resultats_enrichis = self._enrichir_resultats_analyse(
                        resultats_analyse, texte_document, nom_fichier
                    )

                    # Ajouter des métadonnées
                    resultats_enrichis["date_analyse"] = datetime.now().isoformat()
                    resultats_enrichis["methode"] = "ia"

                    # Calculer un hash du document pour traçabilité
                    resultats_enrichis["document_hash"] = hashlib.md5(texte_document.encode('utf-8')).hexdigest()

                    return resultats_enrichis
            except Exception as e:
                logger.warning(f"Erreur lors de l'analyse par IA: {e}")

            # Fallback: utiliser l'approche traditionnelle par mots-clés (méthode originale)
            doc_type = self.classify_document(texte_document)

            return {
                "categorie": doc_type,
                "sous_categorie": "",
                "confiance": 0.7,
                "type_document": self._identifier_type_document(texte_document, nom_fichier),
                "methode": "mots_cles",
                "mots_cles_detectes": self._identifier_mots_cles(texte_document),
            }

        except Exception as e:
            logger.error(f"Erreur lors de la classification du document: {e}")
            traceback.print_exc()
            # Retourner une classification par défaut
            return {
                "categorie": "autre",
                "sous_categorie": "",
                "confiance": 0.3,
                "type_document": "inconnu",
                "methode": "fallback",
                "erreur": str(e),
            }

    def classify_document(self, text: str) -> str:
        """
        Classifie un document en fonction de son contenu textuel (méthode originale)

        Args:
            text: Texte du document

        Returns:
            str: Type de document
        """
        if not text:
            return "autre"

        # Convertir le texte en minuscules
        text = text.lower()

        # Compter les occurrences des mots-clés pour chaque type
        scores = {doc_type: 0 for doc_type in self.keywords.keys()}

        for doc_type, keywords in self.keywords.items():
            for keyword in keywords:
                # Compter les occurrences du mot-clé
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))

                # Certains mots-clés sont plus importants que d'autres
                if keyword in [
                    "facture",
                    "invoice",
                    "contrat",
                    "relevé",
                    "impôt",
                    "assurance",
                    "ordonnance",
                    "passeport",
                    "billet",
                    "bail",
                ]:
                    count *= 2  # Donner plus de poids aux mots-clés importants

                scores[doc_type] += count

        # Rechercher des patterns spécifiques pour certains types de documents
        # Facture
        if re.search(r'facture\s+n[o°]', text) or re.search(r'invoice\s+n[o°]', text):
            scores["facture"] += 5

        # Relevé bancaire
        if re.search(r'relevé\s+de\s+compte', text) or re.search(r'opérations\s+du\s+\d{1,2}/\d{1,2}', text):
            scores["releve_bancaire"] += 5

        # Impôts
        if re.search(r"avis\s+d['\"]impôt", text) or re.search(r"numéro\s+fiscal", text):
            scores["impot"] += 5

        # Santé
        if re.search(r'ordonnance\s+médicale', text) or re.search(r'posologie', text):
            scores["sante"] += 5

        # Identité
        if re.search(r"carte\s+nationale\s+d['\"]identité", text) or re.search(r"passeport", text):
            scores["identite"] += 5

        # Trouver le type avec le score le plus élevé
        max_score = 0
        best_type = "autre"

        for doc_type, score in scores.items():
            if score > max_score:
                max_score = score
                best_type = doc_type

        # Si le score est trop faible, classifier comme "autre"
        if max_score < 2:
            best_type = "autre"

        logger.info(f"Document classifié comme '{best_type}' avec un score de {max_score}")

        # Mappage vers les nouvelles catégories si nécessaire
        mapping = {
            "releve_bancaire": "banque",
            # Autres mappages si nécessaire
        }

        return mapping.get(best_type, best_type)

    def get_document_keywords(self, document_type: str) -> list[str]:
        """
        Récupère les mots-clés associés à un type de document

        Args:
            document_type: Type de document

        Returns:
            List[str]: Liste des mots-clés
        """
        return self.keywords.get(document_type, [])

    def suggerer_classement(self, resultats_analyse: dict[str, Any]) -> dict[str, Any]:
        """
        Suggère comment classer un document

        Args:
            resultats_analyse: Résultats de l'analyse du document

        Returns:
            Dict: Suggestions de classement
        """
        try:
            # Utiliser l'IA pour suggérer un classement si disponible
            try:
                return self.openai_service.document_analyzer.suggerer_classement(resultats_analyse)
            except Exception as e:
                logger.warning(f"Erreur lors de la suggestion de classement via IA: {e}")

            # Suggestions par défaut basées sur la catégorie
            categorie = resultats_analyse.get("categorie", "autre")
            type_document = resultats_analyse.get("type_document", "document")

            # Nommer le fichier
            nom_base = f"{type_document}_{datetime.now().strftime('%Y%m%d')}"

            return {
                "categorie": categorie,
                "sous_categorie": "",
                "nom_fichier": f"{nom_base}.pdf",
                "priorite": "Standard",
                "actions": [],
                "mots_cles": [],
            }

        except Exception as e:
            logger.error(f"Erreur lors de la suggestion de classement: {e}")

            # Fallback simple
            return {
                "categorie": "autre",
                "sous_categorie": "",
                "nom_fichier": f"document_{datetime.now().strftime('%Y%m%d')}.pdf",
                "priorite": "Standard",
                "actions": [],
                "mots_cles": [],
            }

    def extraire_contacts(self, texte_document: str) -> list[dict[str, str]]:
        """
        Extrait les informations de contact d'un document

        Args:
            texte_document: Texte du document

        Returns:
            List[Dict]: Liste des contacts extraits
        """
        try:
            # Utiliser l'IA pour extraire les contacts si disponible
            try:
                return self.openai_service.document_analyzer.extraire_contacts(texte_document)
            except Exception as e:
                logger.warning(f"Erreur lors de l'extraction des contacts via IA: {e}")

            # Méthode de fallback: extraction basique par regex
            contacts = []

            # Rechercher des emails
            emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', texte_document)

            # Rechercher des numéros de téléphone
            telephones = re.findall(
                r'(?:\+\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]?\d{2,4}[- ]?\d{2,4}[- ]?\d{2,4}', texte_document
            )

            # Rechercher des noms d'entreprise (simple heuristique)
            entreprises = re.findall(
                r'(?:SA|SARL|SAS|SNC|EURL|Société|Entreprise|Ets)\s+([A-Z][A-Za-z\s]+(?:&\s+[A-Za-z\s]+)?)',
                texte_document,
            )

            # Si des informations ont été trouvées, créer un contact
            if emails or telephones or entreprises:
                contact = {
                    "nom": entreprises[0] if entreprises else "Contact",
                    "type": "entreprise",
                    "adresse": "",
                    "telephone": telephones[0] if telephones else "",
                    "email": emails[0] if emails else "",
                    "site_web": "",
                    "identifiants": "",
                }
                contacts.append(contact)

            return contacts

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des contacts: {e}")
            return []

    def extraire_informations_cles(self, texte_document: str, type_document: str = "general") -> dict[str, Any]:
        """
        Extrait les informations clés d'un document selon son type

        Args:
            texte_document: Texte du document
            type_document: Type du document

        Returns:
            Dict: Informations clés extraites
        """
        try:
            # Utiliser l'IA pour extraire les informations si disponible
            try:
                # Construire un prompt spécifique au type de document
                if type_document == "facture":
                    prompt = f"""Extrais les informations clés de cette facture:

                    {texte_document[:5000]}

                    Identifie et extrait:
                    1. Numéro de facture
                    2. Date d'émission
                    3. Date d'échéance
                    4. Montant total HT
                    5. Montant TVA
                    6. Montant total TTC
                    7. Émetteur (nom entreprise, adresse)
                    8. Destinataire (nom, adresse)
                    9. Mode de paiement
                    10. Références client/commande

                    Réponds au format JSON avec ces clés.
                    """

                elif type_document == "releve_bancaire":
                    prompt = f"""Extrais les informations clés de ce relevé bancaire:

                    {texte_document[:5000]}

                    Identifie et extrait:
                    1. Banque émettrice
                    2. Numéro de compte
                    3. Nom du titulaire
                    4. Période concernée
                    5. Solde initial
                    6. Solde final
                    7. Total des crédits
                    8. Total des débits
                    9. Principales transactions (liste)

                    Réponds au format JSON avec ces clés.
                    """

                elif type_document == "contrat":
                    prompt = f"""Extrais les informations clés de ce contrat:

                    {texte_document[:5000]}

                    Identifie et extrait:
                    1. Type de contrat
                    2. Parties concernées
                    3. Date de signature
                    4. Date d'entrée en vigueur
                    5. Durée du contrat
                    6. Montants engagés
                    7. Clauses importantes
                    8. Conditions de résiliation

                    Réponds au format JSON avec ces clés.
                    """

                else:
                    # Document général
                    prompt = f"""Extrais les informations clés de ce document:

                    {texte_document[:5000]}

                    Identifie et extrait:
                    1. Type de document
                    2. Date du document
                    3. Émetteur
                    4. Destinataire
                    5. Objet principal
                    6. Montants mentionnés
                    7. Dates importantes
                    8. Actions requises
                    9. Mots-clés importants

                    Réponds au format JSON avec ces clés.
                    """

                # Appeler l'API OpenAI
                response = self.openai_service.demander_reponse(
                    prompt=prompt,
                    system_message="Tu es un expert en extraction d'informations de documents administratifs et financiers.",
                )

                # Extraire et parser la réponse JSON
                try:
                    # Trouver le JSON dans la réponse
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1

                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx]
                        return json.loads(json_str)
                except:
                    pass
            except Exception as e:
                logger.warning(f"Erreur lors de l'extraction via IA: {e}")

            # Méthode de fallback: extraction basique par regex
            informations = {
                "type_document": type_document,
                "date_document": self._extraire_date(texte_document),
                "montants": self._extraire_montants(texte_document),
                "references": self._extraire_references(texte_document),
            }

            return informations

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des informations clés: {e}")
            return {"erreur": str(e)}

    def identifier_actions_necessaires(
        self, texte_document: str, resultats_analyse: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Identifie les actions nécessaires à partir d'un document

        Args:
            texte_document: Texte du document
            resultats_analyse: Résultats de l'analyse

        Returns:
            List[Dict]: Liste des actions nécessaires
        """
        try:
            # Utiliser l'IA pour identifier les actions si disponible
            try:
                # Construire le prompt pour l'IA
                type_doc = resultats_analyse.get("type_document", "général")
                dates = resultats_analyse.get("dates", {})
                montants = resultats_analyse.get("montants", {})

                # Convertir les dates et montants en texte pour le prompt
                dates_str = (
                    ", ".join([f"{k}: {v}" for k, v in dates.items()]) if isinstance(dates, dict) else str(dates)
                )
                montants_str = (
                    ", ".join([f"{k}: {v}" for k, v in montants.items()])
                    if isinstance(montants, dict)
                    else str(montants)
                )

                prompt = f"""Identifie les actions nécessaires à partir de ce document {type_doc}:

                Informations déjà extraites:
                - Type de document: {type_doc}
                - Dates importantes: {dates_str}
                - Montants: {montants_str}

                Texte du document:
                {texte_document[:3000]}

                Pour chaque action identifiée, détermine:
                1. Le type d'action (paiement, réponse, signature, archivage, etc.)
                2. La description de l'action
                3. L'échéance si applicable
                4. La priorité (Urgente, Importante, Normale, Facultative)

                Réponds avec un tableau JSON d'actions nécessaires.
                """

                # Appeler l'API OpenAI
                response = self.openai_service.demander_reponse(
                    prompt=prompt,
                    system_message="Tu es un expert en analyse documentaire et en extraction d'actions nécessaires.",
                )

                # Extraire et parser la réponse JSON
                try:
                    # Trouver le JSON dans la réponse
                    start_idx = response.find('[')
                    end_idx = response.rfind(']') + 1

                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx]
                        return json.loads(json_str)

                    # Si pas de tableau, chercher un objet
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1

                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx]
                        result = json.loads(json_str)

                        # Si l'objet contient une liste d'actions
                        for key in result:
                            if isinstance(result[key], list):
                                return result[key]

                        # Sinon, retourner l'objet dans une liste
                        return [result]
                except:
                    pass
            except Exception as e:
                logger.warning(f"Erreur lors de l'identification des actions via IA: {e}")

            # Méthode de fallback: identification basique
            actions = []

            # Identifier les actions basiques selon le type de document
            type_doc = resultats_analyse.get("type_document", "")

            # Facture à payer
            if type_doc == "facture" and "montant" in str(resultats_analyse):
                date_echeance = self._extraire_date_echeance(texte_document)
                actions.append(
                    {
                        "type": "paiement",
                        "description": "Payer la facture",
                        "echeance": date_echeance,
                        "priorite": "Importante",
                    }
                )

            # Document avec date d'expiration
            if "expire" in texte_document.lower() or "validité" in texte_document.lower():
                date_expiration = self._extraire_date_expiration(texte_document)
                actions.append(
                    {
                        "type": "suivi",
                        "description": "Vérifier la date d'expiration",
                        "echeance": date_expiration,
                        "priorite": "Normale",
                    }
                )

            # Document à archiver
            actions.append(
                {"type": "archivage", "description": "Archiver le document", "echeance": "", "priorite": "Facultative"}
            )

            return actions

        except Exception as e:
            logger.error(f"Erreur lors de l'identification des actions nécessaires: {e}")
            return []

    def _classifier_par_mots_cles(self, texte: str) -> tuple[str, float]:
        """
        Classifie un document par recherche de mots-clés

        Args:
            texte: Texte du document

        Returns:
            Tuple: (catégorie, score de confiance)
        """
        texte_lower = texte.lower()
        scores = {categorie: 0 for categorie in self.categories}

        # Compter les occurrences des mots-clés par catégorie
        for categorie, mots_cles in self.categories.items():
            for mot_cle in mots_cles:
                occurrences = len(re.findall(r'\b' + re.escape(mot_cle) + r'\b', texte_lower))
                scores[categorie] += occurrences

        # Trouver la catégorie avec le score le plus élevé
        categorie_max = max(scores, key=scores.get)

        # Si aucun mot-clé trouvé, catégorie "autre"
        if scores[categorie_max] == 0:
            return "autre", 0.3

        # Calculer un score de confiance (0.0 - 1.0)
        total_occurrences = sum(scores.values())
        confiance = min(scores[categorie_max] / max(total_occurrences, 1), 1.0)

        return categorie_max, confiance

    def _identifier_type_document(self, texte: str, nom_fichier: str = "") -> str:
        """
        Identifie le type de document

        Args:
            texte: Texte du document
            nom_fichier: Nom du fichier

        Returns:
            str: Type de document
        """
        texte_lower = texte.lower()
        nom_lower = nom_fichier.lower()

        # Types de documents courants
        types = {
            "facture": ["facture", "invoice", "quittance", "reçu", "receipt"],
            "releve_bancaire": ["relevé", "extrait", "statement", "compte", "bancaire"],
            "contrat": ["contrat", "contract", "convention", "agreement"],
            "bulletin_salaire": ["salaire", "paie", "rémunération", "salary", "payslip"],
            "impot": ["impôt", "taxe", "tax", "fiscal", "déclaration"],
            "assurance": ["assurance", "police", "insurance", "sinistre", "prime"],
            "identite": ["identité", "passeport", "carte", "permis", "identity"],
            "courrier": ["courrier", "mail", "lettre", "letter", "correspondance"],
        }

        # Détecter les types dans le nom de fichier et le texte
        for type_doc, mots_cles in types.items():
            # Vérifier dans le nom de fichier (prioritaire)
            if any(mot in nom_lower for mot in mots_cles):
                return type_doc

            # Vérifier dans le texte
            if any(mot in texte_lower[:2000] for mot in mots_cles):
                return type_doc

        return "document"  # Type par défaut

    def _identifier_mots_cles(self, texte: str) -> list[str]:
        """
        Identifie les mots-clés importants dans le document

        Args:
            texte: Texte du document

        Returns:
            List[str]: Mots-clés identifiés
        """
        texte_lower = texte.lower()
        mots_cles_trouves = []

        for mot_cle in self.mots_cles_importants:
            if mot_cle in texte_lower:
                mots_cles_trouves.append(mot_cle)

        return mots_cles_trouves

    def _enrichir_resultats_analyse(self, resultats: dict[str, Any], texte: str, nom_fichier: str) -> dict[str, Any]:
        """
        Enrichit les résultats de l'analyse IA avec des informations supplémentaires

        Args:
            resultats: Résultats de l'analyse IA
            texte: Texte du document
            nom_fichier: Nom du fichier

        Returns:
            Dict: Résultats enrichis
        """
        # Ajouter le type de document s'il manque
        if "type_document" not in resultats or not resultats["type_document"]:
            resultats["type_document"] = self._identifier_type_document(texte, nom_fichier)

        # Déterminer la catégorie principale si absente
        if "categorie_suggestion" in resultats:
            cat = resultats["categorie_suggestion"].lower()
            for categorie in self.categories:
                if categorie in cat or any(kw in cat for kw in self.categories[categorie]):
                    resultats["categorie"] = categorie
                    break

            # Fallback
            if "categorie" not in resultats:
                resultats["categorie"] = "autre"
        elif "categorie" not in resultats:
            # Utiliser l'ancienne méthode par mots-clés comme fallback
            resultats["categorie"], confiance = self._classifier_par_mots_cles(texte)
            resultats["confiance"] = confiance

        # Identifier les mots-clés
        if "mots_cles_detectes" not in resultats:
            resultats["mots_cles_detectes"] = self._identifier_mots_cles(texte)

        # Estimer un niveau de confiance si absent
        if "confiance" not in resultats:
            # Plus de champs remplis = plus de confiance
            champs_remplis = sum(1 for v in resultats.values() if v)
            total_champs = len(resultats)
            resultats["confiance"] = min(champs_remplis / total_champs * 1.5, 1.0)

        return resultats

    def _extraire_date(self, texte: str) -> str:
        """
        Extrait une date du texte

        Args:
            texte: Texte du document

        Returns:
            str: Date extraite ou chaîne vide
        """
        # Patterns de date courants
        patterns = [
            r'\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b',  # JJ/MM/AAAA ou JJ.MM.AAAA
            r'\b(\d{1,2}[-]\d{1,2}[-]\d{2,4})\b',  # JJ-MM-AAAA
            r'\b(\d{2,4}[./]\d{1,2}[./]\d{1,2})\b',  # AAAA/MM/JJ
            r'\b(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{2,4})\b',  # JJ mois AAAA
        ]

        for pattern in patterns:
            matches = re.findall(pattern, texte)
            if matches:
                return matches[0]

        return ""

    def _extraire_date_echeance(self, texte: str) -> str:
        """
        Extrait une date d'échéance du texte

        Args:
            texte: Texte du document

        Returns:
            str: Date d'échéance extraite ou chaîne vide
        """
        texte_lower = texte.lower()

        # Chercher des phrases contenant "échéance", "due", "payer avant", etc.
        echeance_contexts = [
            r'(?:date\s+d\'échéance|échéance|payer\s+avant|date\s+limite|due\s+date)[^\n.]*?(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
            r'(?:date\s+d\'échéance|échéance|payer\s+avant|date\s+limite|due\s+date)[^\n.]*?(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{2,4})',
        ]

        for pattern in echeance_contexts:
            matches = re.findall(pattern, texte_lower)
            if matches:
                return matches[0]

        # Si pas trouvé dans un contexte spécifique, chercher n'importe quelle date
        return self._extraire_date(texte)

    def _extraire_date_expiration(self, texte: str) -> str:
        """
        Extrait une date d'expiration du texte

        Args:
            texte: Texte du document

        Returns:
            str: Date d'expiration extraite ou chaîne vide
        """
        texte_lower = texte.lower()

        # Chercher des phrases contenant "expiration", "valide jusqu'au", etc.
        expiration_contexts = [
            r'(?:expire|expiration|valide\s+jusqu\'au|valable\s+jusqu\'au)[^\n.]*?(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
            r'(?:expire|expiration|valide\s+jusqu\'au|valable\s+jusqu\'au)[^\n.]*?(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{2,4})',
        ]

        for pattern in expiration_contexts:
            matches = re.findall(pattern, texte_lower)
            if matches:
                return matches[0]

        return ""

    def _extraire_montants(self, texte: str) -> dict[str, float]:
        """
        Extrait les montants du texte

        Args:
            texte: Texte du document

        Returns:
            Dict: Montants extraits (clé: contexte, valeur: montant)
        """
        texte_lower = texte.lower()
        montants = {}

        # Chercher des montants avec contexte
        contextes = {
            "total": [r'total[^\n.]*?(\d+[.,]\d+)', r'montant\s+total[^\n.]*?(\d+[.,]\d+)'],
            "ht": [r'(?:montant|total)\s+ht[^\n.]*?(\d+[.,]\d+)', r'hors\s+taxe[^\n.]*?(\d+[.,]\d+)'],
            "tva": [r'(?:montant|total)\s+tva[^\n.]*?(\d+[.,]\d+)', r'tva[^\n.]*?(\d+[.,]\d+)'],
            "ttc": [r'(?:montant|total)\s+ttc[^\n.]*?(\d+[.,]\d+)', r'ttc[^\n.]*?(\d+[.,]\d+)'],
        }

        for contexte, patterns in contextes.items():
            for pattern in patterns:
                matches = re.findall(pattern, texte_lower)
                if matches:
                    # Prendre le premier match et convertir en float
                    try:
                        montant_str = matches[0].replace(',', '.')
                        montants[contexte] = float(montant_str)
                        break  # Passer au contexte suivant
                    except ValueError:
                        pass

        # Si aucun montant spécifique trouvé, chercher tous les montants
        if not montants:
            general_matches = re.findall(r'(\d+[.,]\d+)', texte)
            if general_matches:
                try:
                    # Prendre le montant le plus élevé
                    montants_values = [float(m.replace(',', '.')) for m in general_matches]
                    montants["montant"] = max(montants_values)
                except ValueError:
                    pass

        return montants

    def _extraire_references(self, texte: str) -> dict[str, str]:
        """
        Extrait les références du texte

        Args:
            texte: Texte du document

        Returns:
            Dict: Références extraites (clé: type, valeur: référence)
        """
        texte_lower = texte.lower()
        references = {}

        # Chercher des références avec contexte
        contextes = {
            "facture": [
                r'facture\s+(?:n[o°])?[^\n.]*?([A-Za-z0-9-_]{3,})',
                r'n[o°]\s+(?:de\s+)?facture[^\n.]*?([A-Za-z0-9-_]{3,})',
            ],
            "client": [
                r'client\s+(?:n[o°])?[^\n.]*?([A-Za-z0-9-_]{3,})',
                r'n[o°]\s+(?:de\s+)?client[^\n.]*?([A-Za-z0-9-_]{3,})',
            ],
            "commande": [
                r'commande\s+(?:n[o°])?[^\n.]*?([A-Za-z0-9-_]{3,})',
                r'n[o°]\s+(?:de\s+)?commande[^\n.]*?([A-Za-z0-9-_]{3,})',
            ],
            "compte": [
                r'compte\s+(?:n[o°])?[^\n.]*?([A-Za-z0-9-_]{3,})',
                r'n[o°]\s+(?:de\s+)?compte[^\n.]*?([A-Za-z0-9-_]{3,})',
            ],
        }

        for contexte, patterns in contextes.items():
            for pattern in patterns:
                matches = re.findall(pattern, texte_lower)
                if matches:
                    # Prendre le premier match
                    references[contexte] = matches[0]
                    break  # Passer au contexte suivant

        return references


# Singleton pour accès global au service
_service_instance = None


def get_document_classifier():
    """
    Accède à l'instance singleton du service de classification

    Returns:
        DocumentClassifier: Instance du service
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = DocumentClassifier()
    return _service_instance
