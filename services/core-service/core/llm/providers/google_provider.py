"""
Google Gemini Provider with Search and Maps support
"""

import asyncio
import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from ..llm_request import LLMRequest
from ..llm_response import LLMResponse
from .llm_provider_base import LLMProviderBase
from .llm_provider_error import LLMProviderError
from .llm_provider_timeout_error import LLMProviderTimeoutError

logger = logging.getLogger(__name__)


def _get_enable_mock_llm() -> bool:
    """
    Get ENABLE_MOCK_LLM from environment or settings.
    
    Reads from environment variable ENABLE_MOCK_LLM, with fallback to settings
    if available. Defaults to False (production mode).
    
    Note: This function is used as fallback if enable_mock_llm is not passed
    via kwargs from LLMFactory.
    """
    # Try to get from environment first (for backward compatibility)
    env_value = os.getenv("ENABLE_MOCK_LLM")
    if env_value is not None:
        return env_value.lower() == "true"
    
    # Try to get from settings (Pydantic loads .env automatically)
    try:
        import sys
        import os as os_module
        sys.path.insert(0, os_module.path.join(os_module.path.dirname(__file__), '../../../..'))
        from shared.config.settings import get_core_settings
        settings = get_core_settings()
        return settings.enable_mock_llm
    except Exception:
        # Fallback to False if settings not available
        return False


class GoogleProvider(LLMProviderBase):
    """Google Gemini LLM Provider with Search and Maps support"""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__(api_key, **kwargs)

        # Lire enable_mock_llm depuis les settings ou l'environnement
        enable_mock_llm = kwargs.get("enable_mock_llm")
        if enable_mock_llm is None:
            enable_mock_llm = _get_enable_mock_llm()
        
        self.enable_mock_llm = enable_mock_llm

        # Initialiser le client selon le mode (mock ou production)
        print(f"🔧 [INIT] ENABLE_MOCK_LLM = {self.enable_mock_llm}", flush=True)
        print(f"🔧 [INIT] API Key présente: {bool(api_key)}", flush=True)

        if self.enable_mock_llm:
            print("🔧 [INIT] Mode MOCK activé - pas de connexion GCP", flush=True)
            logger.info("🔧 Initialisation en mode MOCK (pas de connexion GCP)")
            self.client = None  # Mock: pas de client réel
        else:
            print("✅ [INIT] Mode PRODUCTION - initialisation client GCP", flush=True)
            logger.info("✅ Initialisation du client GCP Gemini")
            try:
                self.client = genai.Client(api_key=api_key)
                print(f"✅ [INIT] Client GCP créé: {self.client is not None}", flush=True)
            except Exception as e:
                print(f"❌ [INIT] Erreur création client GCP: {e}", flush=True)
                logger.error(f"Erreur création client GCP: {e}")
                raise

        # Available models (2.5 uniquement)
        self.models = {
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
            "gemini-3-flash-preview": "gemini-3-flash-preview"
        }
        
        # Mapping des anciens noms de modèles vers les nouveaux
        self.model_mapping = {
            "gemini-2.5-flash": "gemini-2.5-flash",  # Migration 2.0 -> 2.5
            "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
            "gemini-3-flash-preview": "gemini-3-flash-preview"
        }

        # Default model (comme le POC)
        self.default_model = "gemini-3-flash-preview"

    async def generate(self, request: LLMRequest, max_retries: int = 3) -> LLMResponse:
        """Generate text using Google Gemini with optional Search and Maps
        
        Args:
            request: LLM request
            max_retries: Maximum number of retries for 500 errors (default: 3)
        """
        try:
            self._validate_request(request)

            model_name = request.model or self.default_model
            
            # Mapper les anciens noms de modèles vers les nouveaux
            if model_name in self.model_mapping:
                model_name = self.model_mapping[model_name]
                logger.info(f"Model mapped from {request.model} to {model_name}")
            
            # Si le modèle n'est toujours pas dans la liste, utiliser le default
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not in available models, using default {self.default_model}")
                model_name = self.default_model

            # Prepare content
            if request.system_message:
                full_prompt = f"{request.system_message}\n\n{request.prompt}"
            else:
                full_prompt = request.prompt

            # 🔍 DEBUG MODE: Log prompt complet
            debug_mode = os.getenv("LLM_DEBUG_MODE", "false").lower() == "true"
            if debug_mode:
                logger.info("=" * 80)
                logger.info("🔍 DEBUG MODE - PROMPT COMPLET")
                logger.info("=" * 80)
                logger.info(f"Provider: {self.__class__.__name__}")
                logger.info(f"Model: {model_name}")
                logger.info(f"System Message: {request.system_message[:200] if request.system_message else 'None'}...")
                logger.info(f"Prompt Length: {len(request.prompt)} chars")
                logger.info(f"\n📝 PROMPT COMPLET:\n{full_prompt}")
                logger.info("=" * 80)

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_prompt)]
                )
            ]

            # ✅ NOUVEAU : Configuration des tools
            tools = []
            if request.use_search:
                logger.info("🔍 Enabling Google Search grounding")
                tools.append(types.Tool(google_search=types.GoogleSearch()))

            if request.use_maps:
                logger.info("🗺️ Enabling Google Maps grounding")
                tools.append(types.Tool(google_maps=types.GoogleMaps()))

            # Configuration de génération (comme dans le POC)
            # Valeur par défaut raisonnable pour éviter les coûts excessifs
            # Si max_tokens n'est pas défini, on limite à 8000 tokens au lieu de 65535
            DEFAULT_MAX_TOKENS = 8000
            generate_content_config = types.GenerateContentConfig(
                temperature=request.temperature,
                top_p=0.95,
                max_output_tokens=request.max_tokens or DEFAULT_MAX_TOKENS,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    )
                ],
                tools=tools if tools else None,
            )

            # Generate content avec retry pour les erreurs 500
            loop = asyncio.get_event_loop()
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    response = await loop.run_in_executor(
                        None,
                        self._generate_sync,
                        model_name,
                        contents,
                        generate_content_config
                    )
                    # Succès, sortir de la boucle
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    # Vérifier si c'est une erreur 500 ou 503 de Google (retry)
                    is_retryable = (
                        "500 INTERNAL" in error_str or 
                        "503 UNAVAILABLE" in error_str or 
                        "ServerError" in error_str
                    )
                    
                    if is_retryable:
                        if attempt < max_retries - 1:
                            # Pour 503 (overloaded), attendre plus longtemps
                            if "503" in error_str:
                                wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s pour 503
                            else:
                                wait_time = 2 ** attempt  # 1s, 2s, 4s pour 500
                            
                            logger.warning(
                                f"Google API error (attempt {attempt + 1}/{max_retries}), "
                                f"retrying in {wait_time}s... Error: {error_str[:200]}"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Google API error after {max_retries} attempts: {error_str[:200]}")
                            raise
                    else:
                        # Autre type d'erreur, ne pas retry
                        raise

            # Parse response
            if not response or not response.text:
                logger.warning(
                    "Empty response from Google Gemini - returning fallback placeholder"
                )
                # Retourner un placeholder au lieu de lever une exception
                return LLMResponse(
                    text="[Contenu temporairement indisponible - API Gemini a retourné une réponse vide]",
                    model=mapped_model,
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    },
                    finish_reason="empty_response",
                    provider="google"
                )

            # Extract usage information (gérer les None pour gemini-3-flash-preview)
            usage = None
            if hasattr(response, 'usage_metadata'):
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                total_tokens = getattr(response.usage_metadata, 'total_token_count', 0)

                # Gemini 2.5 Pro peut retourner None pour completion_tokens
                if completion_tokens is None:
                    completion_tokens = 0
                if prompt_tokens is None:
                    prompt_tokens = 0
                if total_tokens is None:
                    total_tokens = 0

                usage = {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(total_tokens)
                }

            # ✅ NOUVEAU : Extraire metadata grounding si présent
            grounding_metadata = None
            if (request.use_search or request.use_maps) and hasattr(response, 'grounding_metadata'):
                grounding_metadata = {
                    "grounding_support": getattr(response.grounding_metadata, 'grounding_support', None),
                    "search_queries": getattr(response.grounding_metadata, 'search_queries', []),
                    "maps_queries": getattr(response.grounding_metadata, 'maps_queries', [])
                }

            # Gérer les cas où les attributs peuvent être None
            candidates = getattr(response, 'candidates', [])
            if candidates is None:
                candidates = []

            safety_ratings = getattr(response, 'safety_ratings', [])
            if safety_ratings is None:
                safety_ratings = []

            return LLMResponse(
                text=response.text,
                provider="google",
                model=model_name,
                usage=usage,
                finish_reason=getattr(response, 'finish_reason', 'stop'),
                metadata={
                    "safety_ratings": safety_ratings,
                    "candidates": len(candidates),
                    "search_enabled": request.use_search,
                    "maps_enabled": request.use_maps,
                    "grounding_metadata": grounding_metadata
                }
            )

        except Exception as e:
            if isinstance(e, LLMProviderError | LLMProviderTimeoutError):
                raise
            raise LLMProviderError(
                f"Generation failed: {str(e)}",
                provider="google"
            ) from e

    def _generate_sync(
        self,
        model_name: str,
        contents: list,
        generate_content_config: types.GenerateContentConfig
    ):
        """Synchronous generation method with new API (comme le POC)"""

        # Choisir entre mode mock et mode production
        if not self.enable_mock_llm and self.client:
            # ==========================================
            # ✅ MODE PRODUCTION : APPEL RÉEL GCP
            # ==========================================
            print(f"🚀 [GCP] Appel réel à GCP Gemini : {model_name}", flush=True)
            print(f"   - Temperature: {generate_content_config.temperature}", flush=True)
            print(f"   - Max tokens: {generate_content_config.max_output_tokens}", flush=True)
            print(f"   - Tools: {generate_content_config.tools}", flush=True)

            logger.info(f"🚀 Appel réel à GCP Gemini : {model_name}")
            logger.info(f"   - Temperature: {generate_content_config.temperature}")
            logger.info(f"   - Max tokens: {generate_content_config.max_output_tokens}")
            logger.info(f"   - Tools: {generate_content_config.tools}")

            response_text = ""
            chunk_count = 0
            last_chunk = None

            try:
                stream = self.client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=generate_content_config
                )

                print("   [GCP] Stream créé, itération...", flush=True)

                for chunk in stream:
                    chunk_count += 1
                    last_chunk = chunk

                    # Log verbeux désactivé pour éviter de polluer les logs
                    # print(f"   [GCP] Chunk {chunk_count}: type={type(chunk)}, has_text={hasattr(chunk, 'text')}", flush=True)

                    if hasattr(chunk, 'text'):
                        chunk_text = chunk.text
                        # print(f"   [GCP] Chunk text: {chunk_text[:100] if chunk_text else 'EMPTY'}...", flush=True)
                        if chunk_text:
                            response_text += chunk_text
                    else:
                        logger.debug("Chunk sans attribut 'text'")

                    logger.debug(f"   Chunk {chunk_count}: {hasattr(chunk, 'text')}, has_text={chunk.text if hasattr(chunk, 'text') else 'N/A'}")

                print(f"✅ [GCP] Response: {chunk_count} chunks, {len(response_text)} caractères", flush=True)
                logger.info(f"✅ GCP Response: {chunk_count} chunks, {len(response_text)} caractères")

                if not response_text:
                    print(f"⚠️ [GCP] GCP a retourné {chunk_count} chunks mais texte vide - utilisation fallback", flush=True)
                    logger.warning(f"⚠️ GCP a retourné {chunk_count} chunks mais texte vide - utilisation fallback")
                    if last_chunk:
                        print(f"   Last chunk: {last_chunk}", flush=True)
                        print(f"   Candidates: {getattr(last_chunk, 'candidates', [])}", flush=True)
                        print(f"   Safety ratings: {getattr(last_chunk, 'safety_ratings', [])}", flush=True)
                        logger.warning(f"   Last chunk: {last_chunk}")
                        logger.warning(f"   Candidates: {getattr(last_chunk, 'candidates', [])}")
                        logger.warning(f"   Safety ratings: {getattr(last_chunk, 'safety_ratings', [])}")
                    
                    # ✅ Utiliser un placeholder au lieu de laisser vide
                    response_text = "[Contenu temporairement indisponible - API Gemini a retourné une réponse vide. Cela peut arriver si le prompt déclenche des filtres de sécurité ou si l'API est surchargée.]"

            except Exception as e:
                logger.error(f"❌ Erreur lors du streaming GCP: {str(e)}")
                raise

            # Créer un objet de réponse avec gestion des None
            class RealResponse:
                def __init__(self, text, last_chunk):
                    self.text = text

                    # Gérer usage_metadata avec valeurs par défaut
                    if last_chunk and hasattr(last_chunk, 'usage_metadata'):
                        usage_meta = last_chunk.usage_metadata
                        # Créer un objet usage_metadata sécurisé
                        class SafeUsageMetadata:
                            def __init__(self, um):
                                self.prompt_token_count = getattr(um, 'prompt_token_count', 0) or 0
                                self.candidates_token_count = getattr(um, 'candidates_token_count', 0) or 0
                                self.total_token_count = getattr(um, 'total_token_count', 0) or 0
                        self.usage_metadata = SafeUsageMetadata(usage_meta)
                    else:
                        self.usage_metadata = None

                    self.safety_ratings = getattr(last_chunk, 'safety_ratings', []) if last_chunk else []
                    self.candidates = getattr(last_chunk, 'candidates', []) if last_chunk else []
                    self.finish_reason = getattr(last_chunk, 'finish_reason', 'stop') if last_chunk else 'stop'
                    self.grounding_metadata = getattr(last_chunk, 'grounding_metadata', None) if last_chunk else None

            return RealResponse(response_text, last_chunk)

        # ==========================================
        # 🔧 MODE MOCK : RÉPONSES FACTICES
        # ==========================================
        logger.info("🔧 MODE MOCK : Génération de réponse factice (GCP désactivé)")

        # Extraire le prompt pour générer une réponse mock contextuelle
        prompt_text = ""
        if contents and len(contents) > 0:
            if hasattr(contents[0], 'parts') and contents[0].parts:
                if hasattr(contents[0].parts[0], 'text'):
                    prompt_text = contents[0].parts[0].text

        # 🆕 MOCK SPÉCIFIQUE POUR LE THÈME baad9ac0-5da7-4c34-9f3e-73352c2cf0ad (Fournisseurs Batteries)
        THEME_BATTERIES_ID = 'baad9ac0-5da7-4c34-9f3e-73352c2cf0ad'
        prompt_lower = prompt_text.lower()
        
        # Détecter le thème batteries par mots-clés spécifiques (car theme_id n'est pas dans le prompt)
        theme_batteries_keywords = [
            'fournisseurs européens', 'fournisseurs europeens',
            'ul2271', 'ul 2271',
            'lfp', 'lto', 'lithium',
            'packs batterie', 'packs batterie',
            'robotique', 'agv', 'amr',
            'leclanché', 'tyva energie', 'varta', 'forsee power', 'saft'
        ]
        theme_batteries_detected = (
            THEME_BATTERIES_ID in prompt_text or 
            'theme_id' in str(prompt_text).lower() or
            any(keyword in prompt_lower for keyword in theme_batteries_keywords)
        )
        
        if theme_batteries_detected:
            # Détecter d'abord les processeurs d'analyzer (processor_number) - PRIORITÉ
            # Les processeurs d'analyzer ont des mots-clés plus spécifiques
            processor_number = None
            
            # Processor 1: Identification et Segmentation Fournisseurs
            if 'identification' in prompt_lower and 'segmentation' in prompt_lower:
                processor_number = 1
            # Processor 2: Conformité Technique et Certification
            elif (('conformité' in prompt_lower or 'conformite' in prompt_lower) and 
                  ('technique' in prompt_lower or 'certification' in prompt_lower)):
                processor_number = 2
            # Processor 3: Capacités Co-développement et Intégration
            elif (('co-développement' in prompt_lower or 'co-developpement' in prompt_lower or 'codeveloppement' in prompt_lower) or
                  (('capacités' in prompt_lower or 'capacites' in prompt_lower) and ('intégration' in prompt_lower or 'integration' in prompt_lower))):
                processor_number = 3
            # Processor 4: Flexibilité et Risques Opérationnels
            elif (('flexibilité' in prompt_lower or 'flexibilite' in prompt_lower) or
                  ('risques' in prompt_lower and ('opérationnels' in prompt_lower or 'operationnels' in prompt_lower or 'opérationnel' in prompt_lower))):
                processor_number = 4
            # Processor 5: Recommandations et Plan d'Action
            elif (('recommandations' in prompt_lower or 'recommandation' in prompt_lower) or
                  ('plan' in prompt_lower and ('action' in prompt_lower or 'd\'action' in prompt_lower))):
                processor_number = 5
            
            # Détecter le prompt_number (captation) seulement si aucun processor_number n'a été détecté
            prompt_number = None
            if not processor_number:
                # Détection par mots-clés dans le prompt (ordre de priorité, plus spécifique)
                # Prompt 1: Identification Fournisseurs Européens (mots-clés spécifiques)
                if (('identifier' in prompt_lower or 'identification' in prompt_lower) and 
                    ('fournisseurs' in prompt_lower or 'fournisseur' in prompt_lower) and 
                    ('européens' in prompt_lower or 'europeens' in prompt_lower or 'european' in prompt_lower) and
                    'segmentation' not in prompt_lower and
                    'production' not in prompt_lower and
                    'capacités' not in prompt_lower and 'capacites' not in prompt_lower):
                    prompt_number = 1
                # Prompt 2: Capacités de Production Détaillées (mots-clés spécifiques)
                elif (('capacités' in prompt_lower or 'capacites' in prompt_lower or 'capacité' in prompt_lower or 'capacite' in prompt_lower) and 
                      ('production' in prompt_lower) and
                      ('détaillées' in prompt_lower or 'detaillees' in prompt_lower or 'détaillée' in prompt_lower or 'detaillee' in prompt_lower or 'détail' in prompt_lower or 'detail' in prompt_lower) and
                      'co-développement' not in prompt_lower and 'co-developpement' not in prompt_lower and
                      'codeveloppement' not in prompt_lower and
                      'intégration' not in prompt_lower and 'integration' not in prompt_lower and
                      'approvisionnement' not in prompt_lower and 'supply chain' not in prompt_lower):
                    prompt_number = 2
                # Prompt 3: Chaîne d'Approvisionnement (mots-clés spécifiques)
                elif (('chaîne' in prompt_lower or 'chaine' in prompt_lower or 'chain' in prompt_lower) and 
                      ('approvisionnement' in prompt_lower or 'supply chain' in prompt_lower or 'supply' in prompt_lower) and
                      'production' not in prompt_lower and
                      'capacités' not in prompt_lower and 'capacites' not in prompt_lower):
                    prompt_number = 3
                # Prompt 4: Expérience Robotique (mots-clés spécifiques)
                elif (('robotique' in prompt_lower or 'robot' in prompt_lower) and 
                      ('expérience' in prompt_lower or 'experience' in prompt_lower or 'expertise' in prompt_lower) and
                      'ul2271' not in prompt_lower and
                      'conformité' not in prompt_lower and 'conformite' not in prompt_lower):
                    prompt_number = 4
                # Prompt 5: Expertise UL2271 (mots-clés spécifiques)
                elif (('ul2271' in prompt_lower or 'ul 2271' in prompt_lower) and 
                      ('expertise' in prompt_lower or 'certification' in prompt_lower) and
                      'conformité' not in prompt_lower and 'conformite' not in prompt_lower and
                      'technique' not in prompt_lower):
                    prompt_number = 5
                # Prompt 6: Santé Financière (mots-clés spécifiques)
                elif (('financière' in prompt_lower or 'financiere' in prompt_lower or 'financier' in prompt_lower or 'financial' in prompt_lower) and 
                      ('santé' in prompt_lower or 'sante' in prompt_lower or 'health' in prompt_lower) and
                      'support' not in prompt_lower and
                      'services' not in prompt_lower):
                    prompt_number = 6
                # Prompt 7: Support et Services (mots-clés spécifiques)
                elif (('support' in prompt_lower) and 
                      ('services' in prompt_lower or 'service' in prompt_lower) and
                      'financière' not in prompt_lower and 'financiere' not in prompt_lower):
                    prompt_number = 7
            
            # Log de debug pour la détection
            logger.debug(f"🔍 Détection mock: prompt_number={prompt_number}, processor_number={processor_number}, prompt_preview={prompt_text[:200] if prompt_text else 'EMPTY'}")
            
            # Charger les réponses depuis les fichiers JSON si disponibles
            if prompt_number or processor_number:
                try:
                    import json
                    import os as os_module
                    
                    # Chemin vers les fichiers JSON dans le conteneur Docker
                    # Essayer d'abord /app/mocks (dans Docker), puis /tmp (local)
                    base_paths = ['/app/mocks', '/tmp', os_module.path.join(os_module.path.dirname(__file__), '../../mocks')]
                    captation_file = None
                    
                    for base_path in base_paths:
                        test_path = os_module.path.join(base_path, 'captation_results.json')
                        if os_module.path.exists(test_path):
                            captation_file = test_path
                            break
                    
                    # Charger les résultats de captation si c'est un prompt
                    response_text = None
                    if prompt_number:
                        if captation_file and os_module.path.exists(captation_file):
                            with open(captation_file, 'r', encoding='utf-8') as f:
                                captation_results = json.load(f)
                            
                            # Chercher la réponse correspondante
                            for key, value in captation_results.items():
                                if value.get('prompt_number') == prompt_number:
                                    response_text = value.get('response', '')
                                    if response_text:
                                        logger.info(f"🔧 MOCK THÈME BATTERIES : Retour réponse prompt {prompt_number} ({value.get('title', 'N/A')}) depuis {captation_file}")
                                        break
                            
                            if not response_text:
                                # Si pas trouvé, utiliser une réponse générique
                                response_text = f"Réponse mock pour le thème batteries, prompt {prompt_number} (réponse absente dans le fichier JSON)"
                        else:
                            # Si le fichier n'existe pas, utiliser une réponse générique
                            response_text = f"Réponse mock pour le thème batteries, prompt {prompt_number} (fichier JSON non disponible dans {base_paths})"
                    
                    # Charger les résultats d'analyzer si c'est un processeur
                    if processor_number and not response_text:
                        logger.info(f"🔍 Recherche processeur {processor_number} dans analyzer_results.json")
                        analyzer_file = None
                        for base_path in base_paths:
                            test_path = os_module.path.join(base_path, 'analyzer_results.json')
                            if os_module.path.exists(test_path):
                                analyzer_file = test_path
                                logger.info(f"✅ Fichier analyzer_results.json trouvé: {analyzer_file}")
                                break
                        
                        if analyzer_file and os_module.path.exists(analyzer_file):
                            with open(analyzer_file, 'r', encoding='utf-8') as f:
                                analyzer_results = json.load(f)
                            
                            logger.info(f"📊 Fichier analyzer_results.json chargé: {len(analyzer_results)} entrées")
                            
                            # Chercher la réponse correspondante
                            found = False
                            for key, value in analyzer_results.items():
                                if value.get('processor_number') == processor_number:
                                    found = True
                                    response_data = value.get('response', {})
                                    # Si response est un dict (JSON structuré), le convertir en string JSON
                                    if isinstance(response_data, dict):
                                        response_text = json.dumps(response_data, ensure_ascii=False, indent=2)
                                    elif isinstance(response_data, str):
                                        # Si c'est déjà une string, l'utiliser directement
                                        response_text = response_data
                                    else:
                                        # Sinon, convertir en string
                                        response_text = str(response_data) if response_data else ''
                                    
                                    if response_text:
                                        logger.info(f"🔧 MOCK THÈME BATTERIES : Retour réponse processeur {processor_number} ({value.get('title', 'N/A')}) depuis {analyzer_file}")
                                        break
                            
                            if not found:
                                logger.warning(f"⚠️ Processeur {processor_number} non trouvé dans analyzer_results.json")
                            if not response_text:
                                # Si pas trouvé, utiliser une réponse générique
                                logger.warning(f"⚠️ Réponse vide pour processeur {processor_number}, utilisation d'une réponse générique")
                                response_text = f"Réponse mock pour le thème batteries, processeur {processor_number} (réponse absente dans le fichier JSON)"
                        else:
                            # Si le fichier n'existe pas, utiliser une réponse générique
                            response_text = f"Réponse mock pour le thème batteries, processeur {processor_number} (fichier JSON non disponible dans {base_paths})"
                        
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement du mock batteries : {e}")
                    num = prompt_number or processor_number
                    num_type = "prompt" if prompt_number else "processeur"
                    response_text = f"Réponse mock pour le thème batteries, {num_type} {num} (erreur: {str(e)})"
            else:
                # Si ni prompt_number ni processor_number n'ont été détectés, utiliser une réponse générique
                response_text = f"Réponse mock pour le thème batteries (prompt/processeur non reconnu: {prompt_text[:100]}...)"
            
            # Créer un objet de réponse simple (identique à la version réelle)
            class SimpleResponse:
                def __init__(self, text):
                    self.text = text
                    self.usage_metadata = None
                    self.safety_ratings = []
                    self.candidates = []
                    self.finish_reason = "stop"
                    self.grounding_metadata = None

            return SimpleResponse(response_text)

        # ============================================
        # 🆕 MOCK THÈME FINANCEMENT STARTUP PARIS
        # ============================================
        startup_keywords = [
            'financement', 'startup', 'prêt', 'incubateur',
            'aide', 'bpifrance', 'deeptech', 'subvention', 'amorçage',
            'levée de fonds', 'levier financier', 'funding'
        ]
        if any(kw in prompt_lower for kw in startup_keywords):
            logger.info("🔍 MOCK: Detected startup financing theme")

            # Détecter génération de CONTENU (worker-llm) vs STRUCTURE (analysis)
            content_generation_keywords = [
                'rédige', 'redige', 'écris', 'ecris', 'génère le contenu',
                'genere le contenu', 'développe', 'developpe', 'section',
                'paragraphe', 'chapitre suivant', 'détaille', 'detaille',
                'produis', 'compose', 'élabore', 'elabore'
            ]

            is_content_generation = any(kw in prompt_lower for kw in content_generation_keywords)

            # Charger le mock depuis le fichier
            response_text = None

            if is_content_generation:
                # Worker LLM génère du contenu Markdown pour une section
                logger.info("  → Type: Content Generation (Markdown)")

                # Générer du Markdown mock basé sur le sujet détecté dans le prompt
                if 'positionnement' in prompt_lower or 'éligibilité' in prompt_lower:
                    response_text = """### Positionnement Stratégique et Éligibilité Innovation

L'évolution fulgurante des capacités des Large Language Models (LLM) a précipité le secteur de la génération documentaire professionnelle vers un point d'inflexion historique. Notre startup se positionne sur cette rupture technologique en combinant LLM + Grounding Web/Maps pour produire des rapports ancrés dans la réalité terrain.

**Proposition de Valeur Unique :**
- Intelligence Contextuelle via Grounding : croisement données internes + signaux externes (Google Search, Maps)
- Architecture Hybride : Gemini Flash 3.0 + RAG propriétaire garantissant précision et pertinence
- Gouvernance et Auditabilité : chaque section générée accompagnée de sources URL traçables

**Éligibilité JEI (Jeune Entreprise Innovante) :**
Notre projet coche tous les critères pour bénéficier du statut JEI avec ses avantages fiscaux :
- Dépenses R&D ≥ 60% des charges (3 ingénieurs ML)
- Âge < 8 ans (création 2026)
- Indépendance capitalistique (100% fondateurs)
- Véritables activités R&D (LLM fine-tuning, RAG)

**Impact Fiscal Estimé (première année) :**
- Exonération cotisations sociales patronales : ~45K€
- Crédit Impôt Recherche (CIR) : 30% des dépenses R&D → ~80K€
- **Total avantages fiscaux : ~125K€**"""

                elif 'incubateur' in prompt_lower or 'accélérateur' in prompt_lower or 'écosystème' in prompt_lower:
                    response_text = """### L'Écosystème Parisien : Incubateurs et Accélérateurs

Paris concentre l'un des écosystèmes DeepTech les plus dynamiques d'Europe. Voici les incubateurs prioritaires pour une startup LLM/Data :

#### **Station F - Programme Founders**
- **Spécialisation** : Scaling startups tech
- **Coût** : Gratuit (sélection sur dossier)
- **Equity demandé** : 0%
- **Avantages** : Accès 3000m² bureaux, réseau 30 VCs résidents (Sequoia, Accel), programmes AI/SaaS
- **KPI** : ~1000 startups hébergées, taux de levée post-programme : 60%

#### **Le Camping (Google for Startups)**
- **Sponsor** : Google
- **Programme** : 6 mois gratuit
- **Perks** : $100K crédits Google Cloud, mentorship Google Brain/DeepMind, accès beta APIs Gemini
- **Fit** : Startups utilisant massivement GCP et APIs Google

#### **Agoranov - Spécialisation Sciences**
- **Focus** : DeepTech scientifique (Sorbonne, CNRS)
- **Coût** : 350€/mois, 0% equity
- **Atouts** : Expertise scientifique (PhD advisors), partenariats recherche, financement maturation jusqu'à 90K€

**Stratégie recommandée** : Postuler simultanément à Station F (scaling) + Le Camping (crédits GCP). Cumulable et synergique."""

                elif 'bpifrance' in prompt_lower or 'subvention' in prompt_lower or 'financement public' in prompt_lower:
                    response_text = """### Financements Publics et Subventions (Bpifrance & État)

#### **Bourse French Tech (ex-Concours I-Lab)**
- **Montant** : 90K€ à 600K€ (subvention non dilutive)
- **Critère** : Innovation technologique issue de la recherche
- **Calendrier** : 2 appels/an (mars, septembre)
- **Taux d'acceptation** : ~15%
- **Livrables attendus** : Dossier scientifique 20 pages, pitch jury experts 15 min, preuve de concept fonctionnel

#### **Prêt Innovation Bpifrance (PI/PIA)**
- **Montant** : 50K€ à 3M€
- **Taux** : 0% si échec du projet (!), sinon 4-6%
- **Sans garantie personnelle**
- **Conditions** : Entreprise < 5 ans, budget R&D > 20% CA, innovation technologique démontrée

#### **Plan de financement optimal Année 1**
```
Subventions Publiques:
- Bourse French Tech : 90K€
- Aide Maturation Région : 30K€
- CIR (Crédit Impôt) : 80K€
Total non-dilutif : 200K€

Quasi-Fonds Propres:
- Prêt Innovation BPI : 150K€
- Prêts d'honneur : 60K€
Total dette souple : 210K€

TOTAL : 410K€ (100% non-dilutif)
```"""

                else:
                    # Contenu générique pour autres sections
                    response_text = f"""### Financement Startup DeepTech Paris

Cette section présente les stratégies de financement adaptées aux startups DeepTech parisiennes spécialisées dans la génération documentaire par LLM.

**Points clés** :
- Positionnement sur un marché en forte croissance ($12Bn+ TAM reporting BI)
- Avantages compétitifs : LLM fine-tuning + Grounding Web/Maps + RAG sécurisé
- Éligibilité aux dispositifs fiscaux innovants (JEI, CIR)
- Accès aux meilleurs incubateurs européens (Station F, Le Camping)
- Financements publics non-dilutifs disponibles (Bpifrance, Région)

**Recommandations actionnables** :
1. Déposer dossier JEI dès création (formulaire 2069-A-SD)
2. Postuler Station F + Le Camping simultanément
3. Préparer dossier Bourse French Tech 3 mois avant deadline
4. Structurer budget R&D pour maximiser CIR (30% des dépenses)

**Avantages fiscaux cumulés première année** : ~125K€"""

                logger.info(f"✅ Mock Markdown généré: {len(response_text)} caractères")

            elif 'outline' in prompt_lower or 'strategic' in prompt_lower:
                # Analysis service génère un outline JSON structuré
                logger.info("  → Type: Strategic Outline (JSON)")
                try:
                    import json
                    base_paths = ['/app/mocks', '/tmp', os.path.join(os.path.dirname(__file__), '../../mocks')]
                    for base_path in base_paths:
                        test_path = os.path.join(base_path, 'analyzer_results.json')
                        if os.path.exists(test_path):
                            with open(test_path, 'r', encoding='utf-8') as f:
                                mock_data = json.load(f)
                            if 'startup_financing_paris' in mock_data:
                                response_data = mock_data['startup_financing_paris'].get('response', {})
                                response_text = json.dumps(response_data, ensure_ascii=False, indent=2)
                                logger.info(f"✅ Mock JSON chargé: {len(response_text)} caractères")
                            break
                except Exception as e:
                    logger.warning(f"Erreur chargement outline: {e}")
                    response_text = "Erreur chargement mock outline"

            elif 'research' in prompt_lower or 'captation' in prompt_lower:
                # Captation service génère un plan de recherche
                logger.info("  → Type: Research Planning (JSON)")
                try:
                    import json
                    base_paths = ['/app/mocks', '/tmp', os.path.join(os.path.dirname(__file__), '../../mocks')]
                    for base_path in base_paths:
                        test_path = os.path.join(base_path, 'captation_results.json')
                        if os.path.exists(test_path):
                            with open(test_path, 'r', encoding='utf-8') as f:
                                mock_data = json.load(f)
                            if 'startup_financing_paris' in mock_data:
                                response_data = mock_data['startup_financing_paris'].get('response', {})
                                response_text = json.dumps(response_data, ensure_ascii=False, indent=2)
                                logger.info(f"✅ Mock research plan chargé: {len(response_text)} caractères")
                            break
                except Exception as e:
                    logger.warning(f"Erreur chargement research: {e}")
                    response_text = "Erreur chargement mock research"

            else:
                # Fallback: contenu Markdown générique
                logger.info("  → Type: Generic Markdown")
                response_text = """### Stratégie de Financement Startup DeepTech

L'accès au financement pour une startup DeepTech parisienne nécessite une approche structurée combinant aides publiques et investissement privé.

**Approche Recommandée** :
- Phase 1 : Maximiser les financements non-dilutifs (Bpifrance, CIR, JEI)
- Phase 2 : Intégrer un incubateur de référence (Station F, Le Camping)
- Phase 3 : Construire traction avant levée Seed

**Budget type Année 1** : 400-600K€ dont 60-70% non-dilutif possible."""

            # Retourner la réponse
            class SimpleResponse:
                def __init__(self, text):
                    self.text = text
                    self.usage_metadata = None
                    self.safety_ratings = []
                    self.candidates = []
                    self.finish_reason = "stop"
                    self.grounding_metadata = None

            return SimpleResponse(response_text)

        # Générer une réponse mock basée sur le prompt
        if "géolocalisation" in prompt_text.lower() or "accessibilité" in prompt_text.lower():
            response_text = """Analyse géolocalisation et accessibilité commerciale - RÉPONSE MOCK

**Localisation Exacte :**
- Adresse : Rue du Ventoux, 59650 Villeneuve-d'Ascq
- Coordonnées GPS : 50.61669000, 3.16664000
- Code postal : 59650
- Zone commerciale : Centre Commercial Auchan V2

**Accessibilité Transport :**
- Voiture : Accès facile via Boulevard de Valmy, parking gratuit de 3050 places
- Transport public : Lignes de bus 13, 18, 32, métro ligne M1 station "Villeneuve D'Ascq Hôtel De Ville"
- Distance arrêt : 4 minutes à pied

**Environnement Commercial :**
- Centres commerciaux : Auchan V2, Heron Parc à proximité
- Flux de passage : Élevé, particulièrement les week-ends

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "chalandise" in prompt_text.lower() or "démographie" in prompt_text.lower():
            response_text = """Zone de Chalandise et Démographie Commerciale - RÉPONSE MOCK

**Zone Primaire (< 15 minutes) :**
- Villeneuve-d'Ascq : 62 342 habitants
- Lille : 236 710 habitants
- Mons-en-Barœul : ~22 567 habitants

**Profil Démographique :**
- Population totale zone primaire : ~320 000 habitants
- Répartition par âges : 18% 0-17 ans, 25% 18-34 ans, 28% 35-54 ans, 29% 55+ ans
- Composition ménages : Familles, couples, personnes seules (présence étudiante importante)

**Potentiel Commercial :**
- Zone à forte densité de population active et jeune
- Présence étudiante importante (45 000+ étudiants)
- Consommation orientée vers la mode et les services

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "concurrentiel" in prompt_text.lower() or "concurrence" in prompt_text.lower():
            response_text = """Environnement Concurrentiel Commercial - RÉPONSE MOCK

**Concurrents Directs (Mode Masculine) :**
1. **Jules** : Mode masculine accessible, cible 25-45 ans
2. **Celio** : Mode masculine classique et décontractée
3. **Zara Men** : Mode tendance, cible 18-35 ans

**Concurrents Indirects :**
- **Decathlon** : Sportwear et mode décontractée
- **Kiabi** : Mode familiale accessible
- **C&A** : Mode familiale large

**Analyse de Positionnement :**
- Positionnement : Mode masculine accessible et tendance
- Différenciation : Collections éco-responsables, adaptation toutes morphologies
- Points forts : Conseil personnalisé, présence centres commerciaux

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "infrastructure" in prompt_text.lower() or "équipements" in prompt_text.lower():
            response_text = """Infrastructure Commerciale et Géographie d'Usage - RÉPONSE MOCK

**Équipements Commerciaux :**
- Centre Commercial Auchan V2 : 200+ commerces, restaurants et services
- Heron Parc : Centre commercial et de loisirs
- Parking : 3050 places gratuites

**Infrastructures :**
- Transport : Métro ligne 1, bus, parking gratuit
- Éducation : Université de Lille, écoles supérieures
- Santé : CHRU Lille, cabinets médicaux
- Loisirs : Cinéma UGC, restaurants, bars

**Géographie Commerciale :**
- Zone commerciale V2 : Cœur de l'activité commerciale
- Zones d'activités : EuraTechnologies, Haute Borne, Parc des Moulins
- Résidentiel : Densité significative autour du centre commercial

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "événements" in prompt_text.lower() or "flux" in prompt_text.lower():
            response_text = """Événements Commerciaux et Flux de Clientèle - RÉPONSE MOCK

**Événements Commerciaux Récurrents :**
- Halloween Circus à Aushopping V2 : Du 22 octobre au 1er novembre 2025
- Marchés de Noël : Playground Market (décembre 2025)
- Jazz à Véd'A : Saison 25/26 (octobre 2025 - mai 2026)

**Flux de Clientèle :**
- Étudiants : 45 000+ étudiants et chercheurs sur les campus
- Saisonnalité : Fortes périodes en août-septembre (rentrée), novembre-décembre (fêtes)
- Affluence : Pic les week-ends et périodes de soldes

**Recommandations Commerciales :**
- Cibler les étudiants : Offres spéciales rentrée, promotions
- Événements locaux : Participation aux marchés de Noël, Halloween
- Partenariats : Universités, résidences étudiantes

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "tendances" in prompt_text.lower() or "innovations" in prompt_text.lower():
            response_text = """Tendances et Innovations Mode Locales - RÉPONSE MOCK

**Tendances Mode Émergentes :**
1. **Mode durable** : Matériaux éco-responsables, production éthique
2. **Mode streetwear** : Continuation des tendances urbaines
3. **Mode vintage** : Retour des styles années 90-2000
4. **Personnalisation** : Adaptation à toutes les morphologies

**Évolutions Comportements d'Achat :**
- E-commerce : Canal essentiel, complément du magasin physique
- Personnalisation : Conseil expert, adaptation morphologie
- Location : Tendance émergente pour mode masculine

**Innovations Services Mode :**
- Applications de style personnel : Conseils d'experts en magasin
- Plateformes de revente : Vinted, Vestiaire Collective (impact limité)

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "climat" in prompt_text.lower() or "saisonnalité" in prompt_text.lower():
            response_text = """Climat et Saisonnalité des Ventes - RÉPONSE MOCK

**Profil Climatique :**
- Climat : Océanique tempéré
- Température moyenne : 10-12°C
- Hiver : 1-4°C (très froid, venteux, nuageux)
- Été : 17-18°C (doux, court)

**Impact sur les Ventes :**
- Automne/Hiver : Forte demande vêtements chauds (manteaux, pulls, bottes)
- Printemps : Transition vers tenues plus légères
- Été : Vêtements légers et aérés

**Saisonnalité des Ventes :**
- Période forte : Octobre-Mars (vêtements chauds)
- Transition : Avril-Mai (mi-saison)
- Saison estivale : Juin-Août (vêtements légers)

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        # ==========================================
        # 🚧 MOCK ANALYZER : Réponses pour les processeurs d'analyse
        # ==========================================
        elif ("contextes" in prompt_text.lower() or "contexte commercial" in prompt_text.lower()) or ("processor" in prompt_text.lower() and ("1" in prompt_text or "contextes" in prompt_text.lower())):
            response_text = """## 1. Positionnement Géographique

Le magasin Jules bénéficie d'une implantation stratégique au sein du Centre Commercial Auchan V2, un pôle majeur de la Métropole Européenne de Lille. Cette localisation offre un accès privilégié à une zone de chalandise étendue grâce à l'excellente desserte en transports en commun et au parking gratuit de 3050 places.

**Avantages :**
- Accessibilité optimale via métro ligne 1, bus et axes autoroutiers (A1, A22, A23)
- Forte densité de population dans la zone primaire (< 15 minutes)
- Présence étudiante importante (45 000+ étudiants)

**Contraintes :**
- Concurrence importante dans le centre commercial
- Variation saisonnière des flux de clientèle

## 2. Potentiel Commercial

Le marché présente un potentiel élevé avec une population de plus de 320 000 habitants dans la zone primaire. La clientèle est diversifiée : jeunes actifs, étudiants, familles, avec un pouvoir d'achat hétérogène mais globalement dynamique.

**Évaluation quantitative :**
- Zone primaire : ~320 000 habitants
- Zone secondaire : Élargissement jusqu'à Roubaix, Tourcoing (population supplémentaire importante)
- Présence étudiante : 45 000+ étudiants (cible privilégiée)

## 3. Concurrence

L'environnement concurrentiel est marqué par la présence de marques similaires (Celio, Zara Men) dans le centre commercial, nécessitant une différenciation claire.

**Positionnement :**
- Mode masculine accessible et tendance
- Collections éco-responsables
- Adaptation à toutes les morphologies
- Conseil personnalisé

## 4. Opportunités

Identification de segments porteurs :
- Clientèle étudiante : Offres ciblées rentrée, promotions
- Événements locaux : Marchés de Noël, Halloween Circus
- Mode durable : Tendance émergente forte

## 5. Risques

Facteurs limitants identifiés :
- Forte pluviométrie annuelle nécessitant stocks adaptés
- Variation saisonnière importante des ventes
- Sensibilité aux promotions et soldes

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "clients" in prompt_text.lower() and ("processor" in prompt_text.lower() or "2" in prompt_text or "segment" in prompt_text.lower()):
            response_text = """{
  "segments": [
    {
      "nom": "Étudiants",
      "poids": 35,
      "profil": "18-25 ans, pouvoir d'achat limité, recherche mode accessible et tendance",
      "besoins": "Pièces polyvalentes, promotions, offres étudiantes",
      "comportement": "Achat lors soldes, rentrée universitaire, événements étudiants"
    },
    {
      "nom": "Jeunes Actifs",
      "poids": 30,
      "profil": "25-35 ans, pouvoir d'achat moyen, recherche équilibre style/prix",
      "besoins": "Tenues professionnelles décontractées, pièces durables",
      "comportement": "Achat régulier, fidélité aux enseignes, sensibilité qualité/prix"
    },
    {
      "nom": "Familles",
      "poids": 20,
      "profil": "35-50 ans, pouvoir d'achat variable, recherche praticité et durabilité",
      "besoins": "Vêtements fonctionnels, adaptés aux activités familiales",
      "comportement": "Achat saisonnier, sensibilité aux promotions, recherche qualité"
    },
    {
      "nom": "Seniors Actifs",
      "poids": 15,
      "profil": "50+ ans, pouvoir d'achat moyen à élevé, recherche confort et classicisme",
      "besoins": "Vêtements adaptés morphologie, coupes classiques",
      "comportement": "Fidélité aux marques, recherche conseil, achat raisonné"
    }
  ],
  "opportunites_croissance": [
    "Développer offres étudiantes ciblées",
    "Renforcer présence événements locaux",
    "Proposer services personnalisés (conseil, retouches)"
  ]
}

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "quoi vendre" in prompt_text.lower() or ("produits" in prompt_text.lower() and "processor" in prompt_text.lower()) or ("3" in prompt_text and "processor" in prompt_text.lower()):
            response_text = """{
  "produits_phares": [
    {
      "categorie": "Vêtements d'hiver",
      "poids": 40,
      "produits": ["Manteaux", "Doudounes", "Pulls épais", "Bottes"],
      "justification": "Climat froid hiver, forte demande octobre-mars"
    },
    {
      "categorie": "Vêtements mi-saison",
      "poids": 30,
      "produits": ["Vestes légères", "Pulls fins", "Bottines"],
      "justification": "Périodes transition printemps/automne importantes"
    },
    {
      "categorie": "Vêtements d'été",
      "poids": 20,
      "produits": ["T-shirts", "Shorts", "Pantalons légers"],
      "justification": "Saison estivale courte mais demande présente"
    },
    {
      "categorie": "Accessoires",
      "poids": 10,
      "produits": ["Écharpes", "Bonnets", "Gants", "Accessoires pluie"],
      "justification": "Climat pluvieux et venteux nécessite accessoires protecteurs"
    }
  ],
  "tendances": [
    "Mode durable : Matériaux éco-responsables",
    "Streetwear : Influence urbaine forte",
    "Personnalisation : Adaptation morphologies"
  ],
  "recommandations": [
    "Renforcer stocks vêtements chauds pour hiver",
    "Proposer collections éco-responsables",
    "Diversifier offre accessoires pluie"
  ]
}

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "combien" in prompt_text.lower() or ("volume" in prompt_text.lower() and "processor" in prompt_text.lower()) or ("4" in prompt_text and "processor" in prompt_text.lower()):
            response_text = """{
  "estimations_ca": {
    "ca_mensuel_moyen": "45 000 - 65 000 €",
    "ca_annuel_estime": "540 000 - 780 000 €",
    "methodologie": "Basé sur surface magasin, positionnement, zone de chalandise"
  },
  "indicateurs_cles": {
    "panier_moyen": "35 - 50 €",
    "frequence_visite": "2-3 fois par mois (clientèle fidèle)",
    "taux_rotation": "4-6 fois par an"
  },
  "variations_saisonnieres": {
    "periode_forte": "Octobre - Mars : +20% vs moyenne",
    "periode_ete": "Juin - Août : -10% vs moyenne",
    "soldes": "Janvier, Juillet : +40% vs moyenne"
  },
  "objectifs_croissance": {
    "objectif_an_1": "+10% CA",
    "objectif_an_2": "+15% CA",
    "leviers": ["Offres étudiantes", "Événements locaux", "E-commerce"]
  }
}

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "comment" in prompt_text.lower() and ("processor" in prompt_text.lower() or "stratégie" in prompt_text.lower() or "5" in prompt_text):
            response_text = """{
  "strategie_vente": {
    "approche": "Mix online/offline avec conseil personnalisé",
    "positionnement": "Mode masculine accessible, tendance, éco-responsable"
  },
  "canaux_distribution": [
    {
      "canal": "Magasin physique",
      "poids": 70,
      "avantages": "Conseil expert, essayage, expérience client"
    },
    {
      "canal": "E-commerce",
      "poids": 30,
      "avantages": "Confort, livraison rapide, complémentarité magasin"
    }
  ],
  "modalites_vente": [
    "Prix fixes avec promotions saisonnières",
    "Programme fidélité",
    "Offres étudiantes (-10%)",
    "Services : retouches, conseils personnalisés"
  ],
  "communication": [
    "Réseaux sociaux : Instagram, Facebook",
    "Partenariats : Universités, événements locaux",
    "Newsletter : Promotions, nouveautés"
  ]
}

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "partenaires" in prompt_text.lower() and ("processor" in prompt_text.lower() or "6" in prompt_text):
            response_text = """{
  "partenaires_strategiques": [
    {
      "type": "Universités",
      "partenaires": ["Université de Lille", "Écoles d'ingénieurs"],
      "actions": ["Offres étudiantes", "Événements campus", "Sponsoring"],
      "impact": "Accès à 45 000+ étudiants, fidélisation jeune clientèle"
    },
    {
      "type": "Événements locaux",
      "partenaires": ["Marchés de Noël", "Halloween Circus", "Jazz à Véd'A"],
      "actions": ["Participation stands", "Promotions événementielles"],
      "impact": "Visibilité locale, accès nouveaux clients"
    },
    {
      "type": "Centres commerciaux",
      "partenaires": ["Aushopping V2", "Heron Parc"],
      "actions": ["Opérations commerciales conjointes", "Communication partagée"],
      "impact": "Flux clientèle mutualisé, synergie commerciale"
    }
  ],
  "opportunites_nouveaux_partenariats": [
    "Résidences étudiantes : Offres exclusives",
    "Associations sportives : Partenariats équipementiers",
    "Influencers locaux : Partenariats visibilité"
  ]
}

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        elif "actions" in prompt_text.lower() and ("processor" in prompt_text.lower() or "7" in prompt_text or "plan d'action" in prompt_text.lower()):
            response_text = """{
  "actions_prioritaires": [
    {
      "priorite": "Haute",
      "action": "Développer programme fidélité étudiant",
      "delai": "1 mois",
      "impact": "Fidélisation clientèle jeune, +15% fréquentation"
    },
    {
      "priorite": "Haute",
      "action": "Renforcer stocks vêtements hiver (octobre-mars)",
      "delai": "2 semaines",
      "impact": "Répondre forte demande saisonnière, +20% CA hiver"
    },
    {
      "priorite": "Moyenne",
      "action": "Participer événements locaux (marchés Noël, Halloween)",
      "delai": "2 mois",
      "impact": "Visibilité locale, nouveaux clients, +10% CA événementiel"
    },
    {
      "priorite": "Moyenne",
      "action": "Lancer collection éco-responsable",
      "delai": "3 mois",
      "impact": "Différenciation concurrentielle, attractivité clientèle consciente"
    }
  ],
  "actions_moyen_terme": [
    "Développer e-commerce avec click & collect",
    "Partenariats universités pour offres exclusives",
    "Services conseil personnalisé renforcé"
  ]
}

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        else:
            # Réponse mock générique
            response_text = f"""Réponse Mock - Développement

Vous avez demandé une analyse concernant : {prompt_text[:100]}...

Cette réponse est générée automatiquement en mode développement sans faire appel à l'API Google Gemini.

**Informations Mock :**
- Modèle utilisé : {model_name}
- Longueur du prompt : {len(prompt_text)} caractères
- Search enabled : {getattr(generate_content_config, 'tools', None) is not None}

Pour obtenir une vraie réponse LLM, activez l'appel réel à GCP dans le code.

⚠️ **NOTE : Cette réponse est un mock généré en mode développement. Le vrai appel à GCP est désactivé.**"""

        # Créer un objet de réponse simple (identique à la version réelle)
        class SimpleResponse:
            def __init__(self, text):
                self.text = text
                self.usage_metadata = None
                self.safety_ratings = []
                self.candidates = []
                self.finish_reason = "stop"
                self.grounding_metadata = None

        return SimpleResponse(response_text)

    async def health_check(self) -> bool:
        """Check if Google Gemini is accessible"""
        try:
            # Simple health check with minimal request
            test_request = LLMRequest(
                prompt="Hello",
                provider="google",
                model="gemini-1.5-flash",
                max_tokens=10
            )

            # Set timeout for health check
            response = await asyncio.wait_for(
                self.generate(test_request),
                timeout=10.0
            )

            return bool(response.text)

        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Get available Google Gemini models"""
        return list(self.models.keys())
