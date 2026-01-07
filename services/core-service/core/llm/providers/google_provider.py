"""
Google Gemini Provider with Search and Maps support
"""

import asyncio
import logging
from typing import Any

from google import genai
from google.genai import types

from ..llm_request import LLMRequest
from ..llm_response import LLMResponse
from .llm_provider_base import LLMProviderBase
from .llm_provider_error import LLMProviderError
from .llm_provider_timeout_error import LLMProviderTimeoutError

logger = logging.getLogger(__name__)


class GoogleProvider(LLMProviderBase):
    """Google Gemini LLM Provider with Search and Maps support"""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__(api_key, **kwargs)

        # Initialize GCP client
        logger.info("Initializing Google Gemini client")
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info("Google Gemini client created successfully")
        except Exception as e:
            logger.error(f"Failed to create Google Gemini client: {e}")
            raise LLMProviderError(
                f"Failed to initialize Google Gemini client: {str(e)}",
                provider="google"
            ) from e

        # Available models (Gemini 2.5 and 3)
        self.models = {
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
            "gemini-3-flash-preview": "gemini-3-flash-preview"
        }

        # Model mapping (for backwards compatibility)
        self.model_mapping = {
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
            "gemini-3-flash-preview": "gemini-3-flash-preview"
        }

        # Default model
        self.default_model = "gemini-3-flash-preview"

    async def generate(self, request: LLMRequest, max_retries: int = 3) -> LLMResponse:
        """
        Generate text using Google Gemini with optional Search and Maps

        Args:
            request: LLM request
            max_retries: Maximum number of retries for 500 errors (default: 3)
        """
        try:
            self._validate_request(request)

            model_name = request.model or self.default_model

            # Map old model names to new ones
            if model_name in self.model_mapping:
                model_name = self.model_mapping[model_name]
                logger.info(f"Model mapped from {request.model} to {model_name}")

            # If model still not in list, use default
            if model_name not in self.models:
                logger.warning(
                    f"Model {model_name} not in available models, using default {self.default_model}"
                )
                model_name = self.default_model

            # Prepare content
            if request.system_message:
                full_prompt = f"{request.system_message}\n\n{request.prompt}"
            else:
                full_prompt = request.prompt

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_prompt)]
                )
            ]

            # Configure tools (Search and Maps)
            tools = []
            if request.use_search:
                logger.info("Enabling Google Search grounding")
                tools.append(types.Tool(google_search=types.GoogleSearch()))

            if request.use_maps:
                logger.info("Enabling Google Maps grounding")
                tools.append(types.Tool(google_maps=types.GoogleMaps()))

            # Generation configuration
            # Default max tokens to 8000 to avoid excessive costs
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

            # Generate content with retry for 500 errors
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
                    # Success, break out of loop
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    # Check if it's a retryable 500 or 503 error from Google
                    is_retryable = (
                        "500 INTERNAL" in error_str or
                        "503 UNAVAILABLE" in error_str or
                        "ServerError" in error_str
                    )

                    if is_retryable:
                        if attempt < max_retries - 1:
                            # For 503 (overloaded), wait longer
                            if "503" in error_str:
                                wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s for 503
                            else:
                                wait_time = 2 ** attempt  # 1s, 2s, 4s for 500

                            logger.warning(
                                f"Google API error (attempt {attempt + 1}/{max_retries}), "
                                f"retrying in {wait_time}s... Error: {error_str[:200]}"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(
                                f"Google API error after {max_retries} attempts: {error_str[:200]}"
                            )
                            raise
                    else:
                        # Other type of error, don't retry
                        raise

            # Parse response
            if not response or not response.text:
                logger.warning(
                    "Empty response from Google Gemini - returning fallback placeholder"
                )
                # Return placeholder instead of raising exception
                return LLMResponse(
                    text="[Content temporarily unavailable - Gemini API returned empty response]",
                    provider="google",
                    model=model_name,
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    },
                    finish_reason="empty_response",
                    metadata={}
                )

            # Extract usage information (handle None for gemini-3-flash-preview)
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
                total_tokens = getattr(response.usage_metadata, 'total_token_count', 0) or 0

                usage = {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(total_tokens)
                }

            # Extract grounding metadata if present
            grounding_metadata = None
            if (request.use_search or request.use_maps) and hasattr(response, 'grounding_metadata'):
                grounding_metadata = {
                    "grounding_support": getattr(response.grounding_metadata, 'grounding_support', None),
                    "search_queries": getattr(response.grounding_metadata, 'search_queries', []),
                    "maps_queries": getattr(response.grounding_metadata, 'maps_queries', [])
                }

            # Handle cases where attributes may be None
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
        """Synchronous generation method using Google Gemini API"""

        logger.info(f"Calling Google Gemini API: {model_name}")
        logger.debug(f"Temperature: {generate_content_config.temperature}")
        logger.debug(f"Max tokens: {generate_content_config.max_output_tokens}")
        logger.debug(f"Tools: {generate_content_config.tools}")

        response_text = ""
        chunk_count = 0
        last_chunk = None

        try:
            stream = self.client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config
            )

            logger.debug("Stream created, iterating...")

            for chunk in stream:
                chunk_count += 1
                last_chunk = chunk

                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                    if chunk_text:
                        response_text += chunk_text
                else:
                    logger.debug("Chunk without text attribute")

            logger.info(f"Response: {chunk_count} chunks, {len(response_text)} characters")

            if not response_text:
                logger.warning(
                    f"Gemini returned {chunk_count} chunks but empty text - using fallback"
                )
                if last_chunk:
                    logger.warning(f"Last chunk: {last_chunk}")
                    logger.warning(f"Candidates: {getattr(last_chunk, 'candidates', [])}")
                    logger.warning(f"Safety ratings: {getattr(last_chunk, 'safety_ratings', [])}")

                # Use placeholder instead of leaving empty
                response_text = (
                    "[Content temporarily unavailable - Gemini API returned empty response. "
                    "This can happen if the prompt triggers safety filters or if the API is overloaded.]"
                )

        except Exception as e:
            logger.error(f"Error during Gemini streaming: {str(e)}")
            raise

        # Create response object with None handling
        class GeminiResponse:
            def __init__(self, text, last_chunk):
                self.text = text

                # Handle usage_metadata with default values
                if last_chunk and hasattr(last_chunk, 'usage_metadata'):
                    usage_meta = last_chunk.usage_metadata

                    # Create safe usage_metadata object
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

        return GeminiResponse(response_text, last_chunk)

    async def health_check(self) -> bool:
        """Check if Google Gemini is accessible"""
        try:
            # Simple health check with minimal request
            test_request = LLMRequest(
                prompt="Hello",
                provider="google",
                model="gemini-2.5-flash-lite",  # Use fastest, cheapest model
                max_tokens=10
            )

            # Set timeout for health check
            response = await asyncio.wait_for(
                self.generate(test_request),
                timeout=10.0
            )

            return bool(response.text)

        except Exception as e:
            logger.warning(f"Google Gemini health check failed: {e}")
            return False

    def get_available_models(self) -> list[str]:
        """Get available Google Gemini models"""
        return list(self.models.keys())
