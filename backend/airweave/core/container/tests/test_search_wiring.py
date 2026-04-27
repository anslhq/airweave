"""Unit tests for search-service factory wiring."""

from unittest.mock import MagicMock, patch

from airweave.adapters.llm.fallback import FallbackChainLLM
from airweave.core.container.factory import _build_llm_chain, _create_search_services
from airweave.domains.search.config import SearchConfig


class _ProviderStub:
    def __init__(self, model_spec, max_retries=None) -> None:
        self.model_spec = model_spec
        self.max_retries = max_retries


def _make_settings(**overrides) -> MagicMock:
    defaults = {
        "ANTHROPIC_API_KEY": None,
        "COHERE_API_KEY": None,
        "OPENAI_COMPAT_API_KEY": "compat-key",
        "OPENAI_COMPAT_BASE_URL": "http://localhost:8317/v1",
        "TOGETHER_API_KEY": None,
        "TOGETHER_BASE_URL": None,
        "ENABLE_LOCAL_RERANKER": False,
        "VESPA_URL": "http://localhost",
        "VESPA_PORT": 8081,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for key, value in defaults.items():
        setattr(settings, key, value)
    return settings


def _common_search_service_kwargs(settings: MagicMock) -> dict:
    return {
        "settings": settings,
        "circuit_breaker": MagicMock(),
        "dense_embedder": MagicMock(),
        "sparse_embedder": MagicMock(),
        "collection_repo": MagicMock(),
        "sc_repo": MagicMock(),
        "source_registry": MagicMock(),
        "entity_definition_registry": MagicMock(),
        "event_bus": MagicMock(),
        "source_lifecycle": MagicMock(),
    }


class TestBuildLlmChain:
    def test_openai_compat_provider_selected_explicitly(self):
        settings = _make_settings()
        circuit_breaker = MagicMock()

        with patch("airweave.core.container.factory.OpenAICompatLLM", _ProviderStub):
            llm = _build_llm_chain(settings, SearchConfig(), circuit_breaker)

        assert isinstance(llm, _ProviderStub)
        assert llm.model_spec.api_model_name == "gpt-5.4-mini"

    def test_multiple_available_providers_build_fallback_chain(self):
        settings = _make_settings(ANTHROPIC_API_KEY="anthropic-key")
        circuit_breaker = MagicMock()

        with (
            patch("airweave.core.container.factory.OpenAICompatLLM", _ProviderStub),
            patch("airweave.core.container.factory.AnthropicLLM", _ProviderStub),
        ):
            llm = _build_llm_chain(settings, SearchConfig(), circuit_breaker)

        assert isinstance(llm, FallbackChainLLM)
        assert len(llm._providers) == 2
        assert all(isinstance(provider, _ProviderStub) for provider in llm._providers)


class TestCreateSearchServices:
    def test_reranker_disabled_by_default(self):
        settings = _make_settings()
        llm_stub = MagicMock()

        with (
            patch("airweave.core.container.factory._build_llm_chain", return_value=llm_stub),
            patch("airweave.core.container.factory.CollectionMetadataBuilder"),
            patch("airweave.core.container.factory.VespaVectorDB"),
            patch("vespa.application.Vespa"),
        ):
            deps = _create_search_services(**_common_search_service_kwargs(settings))

        assert deps["classic_search"]._reranker is None
        assert deps["agentic_search"]._reranker is None

    def test_local_reranker_is_opt_in(self):
        settings = _make_settings(ENABLE_LOCAL_RERANKER=True)
        llm_stub = MagicMock()
        reranker_stub = object()

        with (
            patch("airweave.core.container.factory._build_llm_chain", return_value=llm_stub),
            patch("airweave.core.container.factory.CollectionMetadataBuilder"),
            patch("airweave.core.container.factory.VespaVectorDB"),
            patch("vespa.application.Vespa"),
            patch(
                "airweave.core.container.factory.CrossEncoderReranker",
                return_value=reranker_stub,
            ) as cross_encoder_ctor,
        ):
            deps = _create_search_services(**_common_search_service_kwargs(settings))

        assert deps["classic_search"]._reranker is reranker_stub
        cross_encoder_ctor.assert_called_once_with()

    def test_cohere_reranker_takes_precedence(self):
        settings = _make_settings(COHERE_API_KEY="cohere-key", ENABLE_LOCAL_RERANKER=True)
        llm_stub = MagicMock()
        reranker_stub = object()

        with (
            patch("airweave.core.container.factory._build_llm_chain", return_value=llm_stub),
            patch("airweave.core.container.factory.CollectionMetadataBuilder"),
            patch("airweave.core.container.factory.VespaVectorDB"),
            patch("vespa.application.Vespa"),
            patch(
                "airweave.core.container.factory.CohereReranker",
                return_value=reranker_stub,
            ) as cohere_ctor,
            patch("airweave.core.container.factory.CrossEncoderReranker") as cross_encoder_ctor,
        ):
            deps = _create_search_services(**_common_search_service_kwargs(settings))

        assert deps["classic_search"]._reranker is reranker_stub
        cohere_ctor.assert_called_once_with(api_key="cohere-key")
        cross_encoder_ctor.assert_not_called()
