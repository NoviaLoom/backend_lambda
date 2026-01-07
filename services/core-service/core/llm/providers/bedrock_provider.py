"""
AWS Bedrock Provider

Supports:
- Anthropic Claude 3 (Sonnet, Haiku, Opus)
- Meta Llama 3 (8B, 70B)
- Amazon Titan (Text Lite, Text Express)
"""

import asyncio
import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

from ..llm_request import LLMRequest
from ..llm_response import LLMResponse
from .llm_provider_base import LLMProviderBase
from .llm_provider_error import LLMProviderError
from .llm_provider_quota_exceeded_error import LLMProviderQuotaExceededError
from .llm_provider_rate_limit_error import LLMProviderRateLimitError
from .llm_provider_timeout_error import LLMProviderTimeoutError

logger = logging.getLogger(__name__)


class BedrockProvider(LLMProviderBase):
    """AWS Bedrock LLM Provider"""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """
        Initialize Bedrock provider

        Args:
            api_key: Not used for Bedrock (IAM role authentication)
            **kwargs: Additional config (aws_region, etc.)
        """
        super().__init__(api_key, **kwargs)

        # Get AWS region from config or use default
        aws_region = kwargs.get("aws_region", "us-east-1")

        # Initialize Bedrock client (uses IAM role, no explicit credentials)
        try:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=aws_region
            )
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise LLMProviderError(
                f"Bedrock client initialization failed: {str(e)}",
                provider="bedrock"
            ) from e

        # Available models by family
        self.models = {
            # Claude 3 models (Anthropic)
            "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",

            # Llama 3 models (Meta)
            "llama3-8b": "meta.llama3-8b-instruct-v1:0",
            "llama3-70b": "meta.llama3-70b-instruct-v1:0",

            # Amazon Titan models
            "titan-text-lite": "amazon.titan-text-lite-v1",
            "titan-text-express": "amazon.titan-text-express-v1",
        }

        # Default model (Claude 3 Haiku - fastest, most economical)
        self.default_model = "claude-3-haiku"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using AWS Bedrock"""
        try:
            self._validate_request(request)

            # Select model
            model_key = request.model or self.default_model
            model_id = self.models.get(model_key, self.models[self.default_model])

            # Build request body based on model family
            request_body = self._build_request_body(request, model_id)

            # Generate with timeout
            try:
                response = await asyncio.wait_for(
                    self._invoke_model(model_id, request_body),
                    timeout=60.0
                )
            except TimeoutError as e:
                raise LLMProviderTimeoutError(
                    "Request timed out",
                    provider="bedrock"
                ) from e

            # Parse response based on model family
            parsed_response = self._parse_response(response, model_id)

            return LLMResponse(
                text=parsed_response["text"],
                provider="bedrock",
                model=model_key,
                usage=parsed_response.get("usage"),
                finish_reason=parsed_response.get("finish_reason"),
                request_id=response.get("ResponseMetadata", {}).get("RequestId"),
                metadata={
                    "model_id": model_id,
                    "region": self.client.meta.region_name,
                    **parsed_response.get("metadata", {})
                }
            )

        except ClientError as e:
            # Map AWS errors to framework exceptions
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if error_code in ("ThrottlingException", "TooManyRequestsException"):
                raise LLMProviderRateLimitError(
                    f"Rate limit exceeded: {error_message}",
                    provider="bedrock"
                ) from e
            elif error_code in ("ServiceQuotaExceededException", "QuotaExceededException"):
                raise LLMProviderQuotaExceededError(
                    f"Quota exceeded: {error_message}",
                    provider="bedrock"
                ) from e
            else:
                raise LLMProviderError(
                    f"AWS Bedrock error ({error_code}): {error_message}",
                    provider="bedrock"
                ) from e

        except Exception as e:
            if isinstance(e, (LLMProviderError, LLMProviderTimeoutError,
                            LLMProviderRateLimitError, LLMProviderQuotaExceededError)):
                raise
            raise LLMProviderError(
                f"Generation failed: {str(e)}",
                provider="bedrock"
            ) from e

    async def health_check(self) -> bool:
        """Check if Bedrock is accessible"""
        try:
            # Simple health check with minimal request
            test_request = LLMRequest(
                prompt="Hello",
                provider="bedrock",
                model="claude-3-haiku",  # Cheapest model for health check
                max_tokens=10
            )

            # Set timeout for health check
            response = await asyncio.wait_for(
                self.generate(test_request),
                timeout=10.0
            )

            return bool(response.text)

        except Exception as e:
            logger.warning(f"Bedrock health check failed: {e}")
            return False

    def get_available_models(self) -> list[str]:
        """Get available Bedrock models"""
        return list(self.models.keys())

    async def _invoke_model(self, model_id: str, request_body: dict) -> dict:
        """
        Invoke Bedrock model (async wrapper for boto3)

        Args:
            model_id: Bedrock model identifier
            request_body: Request body as dict

        Returns:
            Response dict from Bedrock
        """
        loop = asyncio.get_event_loop()

        def _invoke() -> dict:
            """Synchronous invoke call"""
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )

            # Parse response body
            response_body = json.loads(response["body"].read())

            # Include ResponseMetadata in response
            return {
                **response_body,
                "ResponseMetadata": response.get("ResponseMetadata", {})
            }

        return await loop.run_in_executor(None, _invoke)

    def _build_request_body(self, request: LLMRequest, model_id: str) -> dict:
        """
        Build request body based on model family

        Args:
            request: LLM request
            model_id: Bedrock model ID

        Returns:
            Request body dict
        """
        # Detect model family
        if "claude" in model_id:
            return self._build_claude_request(request)
        elif "llama" in model_id:
            return self._build_llama_request(request)
        elif "titan" in model_id:
            return self._build_titan_request(request)
        else:
            raise LLMProviderError(
                f"Unknown model family: {model_id}",
                provider="bedrock"
            )

    def _build_claude_request(self, request: LLMRequest) -> dict:
        """Build request for Claude 3 models"""
        messages = []

        # Add user message
        messages.append({
            "role": "user",
            "content": request.prompt
        })

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
        }

        # Add system message if provided
        if request.system_message:
            body["system"] = request.system_message

        return body

    def _build_llama_request(self, request: LLMRequest) -> dict:
        """Build request for Llama 3 models"""
        # Llama uses instruction format
        prompt_text = request.prompt
        if request.system_message:
            prompt_text = f"System: {request.system_message}\n\nUser: {prompt_text}"

        return {
            "prompt": prompt_text,
            "max_gen_len": request.max_tokens or 2048,
            "temperature": request.temperature,
            "top_p": 0.9,
        }

    def _build_titan_request(self, request: LLMRequest) -> dict:
        """Build request for Amazon Titan models"""
        text_generation_config = {
            "maxTokenCount": request.max_tokens or 4096,
            "temperature": request.temperature,
            "topP": 0.9,
        }

        # Titan doesn't have explicit system message, prepend to prompt
        prompt_text = request.prompt
        if request.system_message:
            prompt_text = f"{request.system_message}\n\n{prompt_text}"

        return {
            "inputText": prompt_text,
            "textGenerationConfig": text_generation_config
        }

    def _parse_response(self, response: dict, model_id: str) -> dict:
        """
        Parse response based on model family

        Args:
            response: Raw response from Bedrock
            model_id: Model ID used

        Returns:
            Parsed response dict with text, usage, etc.
        """
        # Detect model family
        if "claude" in model_id:
            return self._parse_claude_response(response)
        elif "llama" in model_id:
            return self._parse_llama_response(response)
        elif "titan" in model_id:
            return self._parse_titan_response(response)
        else:
            raise LLMProviderError(
                f"Unknown model family: {model_id}",
                provider="bedrock"
            )

    def _parse_claude_response(self, response: dict) -> dict:
        """Parse Claude 3 response"""
        if not response.get("content"):
            raise LLMProviderError(
                "Empty response from Claude",
                provider="bedrock"
            )

        # Extract text from content array
        text = ""
        for content_block in response["content"]:
            if content_block.get("type") == "text":
                text += content_block.get("text", "")

        # Extract usage
        usage = None
        if "usage" in response:
            usage = {
                "prompt_tokens": response["usage"].get("input_tokens", 0),
                "completion_tokens": response["usage"].get("output_tokens", 0),
                "total_tokens": (
                    response["usage"].get("input_tokens", 0) +
                    response["usage"].get("output_tokens", 0)
                )
            }

        return {
            "text": text,
            "usage": usage,
            "finish_reason": response.get("stop_reason"),
            "metadata": {
                "model": response.get("model"),
                "stop_reason": response.get("stop_reason"),
            }
        }

    def _parse_llama_response(self, response: dict) -> dict:
        """Parse Llama 3 response"""
        if not response.get("generation"):
            raise LLMProviderError(
                "Empty response from Llama",
                provider="bedrock"
            )

        text = response["generation"]

        # Extract usage (Llama format)
        usage = None
        if "prompt_token_count" in response or "generation_token_count" in response:
            prompt_tokens = response.get("prompt_token_count", 0)
            completion_tokens = response.get("generation_token_count", 0)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }

        return {
            "text": text,
            "usage": usage,
            "finish_reason": response.get("stop_reason"),
            "metadata": {
                "stop_reason": response.get("stop_reason"),
            }
        }

    def _parse_titan_response(self, response: dict) -> dict:
        """Parse Amazon Titan response"""
        results = response.get("results", [])
        if not results:
            raise LLMProviderError(
                "Empty response from Titan",
                provider="bedrock"
            )

        text = results[0].get("outputText", "")

        # Extract usage (Titan format)
        usage = None
        if "inputTextTokenCount" in response:
            input_tokens = response.get("inputTextTokenCount", 0)
            completion_tokens = results[0].get("tokenCount", 0)
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": input_tokens + completion_tokens
            }

        return {
            "text": text,
            "usage": usage,
            "finish_reason": results[0].get("completionReason"),
            "metadata": {
                "completion_reason": results[0].get("completionReason"),
            }
        }
