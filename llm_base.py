import llm
import logging
import re

from jinja2 import Environment, FileSystemLoader
from jinja2.environment import Template
from numpy import array, ndarray
from pathlib import Path
from protocols import LanguageModel, LanguageModelLogger, LanguageModelResponse
from typing import Any

logger = logging.getLogger(__name__)


def _validate_options(
    model: llm.Model, options: dict[str, float | int]
) -> dict[str, float | int]:
    """Return the value of each option supported by the model.

    Merge the given options with the default model options, ignoring options
    that are not supported by the model.
    """
    opts: dict[str, float | int] = {}

    for opt_name, value in model.Options.model_fields.items():
        if opt_name in options:
            opts[opt_name] = options[opt_name]
        elif value.default is not None:
            opts[opt_name] = value.default

    extra = set(options) - set(opts)

    if extra:
        logger.warning(
            "The following options are not supported "
            "by the model %s and will be ignored: %s.",
            model.model_id,
            extra,
        )

    return opts


def _call_llm(
    conversation: llm.models.Conversation,
    prompt: str,
    system: str | None,
    options: dict[str, float | int],
) -> llm.models.Response:
    """Low-level function to interact with a language model."""
    response = conversation.prompt(
        prompt=prompt,
        system=system,
        **(options),  # type: ignore
    )

    try:  # to contact the language model
        response.text()
    except Exception as e:
        raise RuntimeError(f"Error contacting {conversation.model.model_id}: {e}")

    return response


class NoOpLogger:  # Conforms to LanguageModelLogger
    def response(self, response: LanguageModelResponse) -> None:
        pass


class PromptTemplate:  # Conforms to LanguageModelPromptTemplate
    """A Jinja prompt template."""

    def __init__(self, template: Template) -> None:
        self.template = template

    @classmethod
    def load(cls, path: str | Path) -> PromptTemplate:
        """Load a template from a file.

        Parameters
        ----------
        path: str | Path
            The template path.
        """
        template = Environment(
            loader=FileSystemLoader(Path(path).parent.absolute())
        ).get_template(Path(path).name)

        return cls(template)

    def apply(self, values: dict[str, Any]) -> str:
        """Instantiate the template with the specified values.

        Parameters
        ----------
        values: dict[str, float | int| str]
            The mapping from the placeholder names to the corresponding values
            to substitute.

        Returns
        -------
        str
            The instantiated template.
        """
        return self.template.render(**values)


class Response:  # Conforms to LanguageModelResponse
    """The response to a prompt.

    This object is not typically instantiated by the user, but it is
    automatically created when prompting a language model.

    Example:

        import llm

        model = llm.get_model("qwen3")
        conversation = model.conversation()  # We need a conversation ID
        prompt = "..."
        llm_response = conversation.prompt(prompt)
        llm_response.text()  # Submit prompt to the LLM
        response = Response(llm_response)

    Parameters
    ----------
    response: llm.models.Response
        The llm response object **after** the language model has been called.
    round: int
        Round of the conversation.
    role: str, default is "default"
        The role of the model in the conversation. For example: "attacker",
        "evaluator", "judge", etc.
    """

    def __init__(
        self, response: llm.models.Response, round: int, role: str = "default"
    ) -> None:
        if response.conversation and hasattr(response.conversation, "id"):
            self.conversation_id: str = response.conversation.id
        else:
            raise ValueError("Response class unexpectedly missing 'id' attribute")

        self.model = response.model.model_id
        self.role = role
        self.round = round
        self._prompt = response.prompt.prompt
        self._system = response.prompt.system
        self._raw_response = response.text()

        # TODO: consider various ways a reasoning block may be defined
        match = re.search(
            r"<think>(.*?)(?:</think>|$)(.*)",
            self._raw_response,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match is None:
            self._reasoning = ""
            self._response = self._raw_response
        else:
            self._reasoning = match[1].strip()
            self._response = match[2].strip()

    def raw_response(self) -> str:
        """The raw response including reasoning."""
        return self._raw_response

    def system_prompt(self) -> str:
        """The system prompt used together with the prompt."""
        return self._system

    def prompt(self) -> str:
        """The prompt that elicited the response."""
        return self._prompt

    def text(self) -> str:
        """The response language model, without the thinking block."""
        return self._response

    def reasoning(self) -> str:
        """The reasoning part of the response.

        Returns
        -------
        str
            The reasoning of the language model, or an empty string if the model
            has no reasoning capabilities.
        """
        return self._reasoning

    def __str__(self) -> str:
        return f"USER: {self._prompt}\nASSISTANT: {self._response}"


class Conversation:  # Conforms to LanguageModel
    """A conversation with an LLM.

    Instances of this class are typically not instantiated directly, but only as
    part of `LLM` objects.

    Parameters
    ----------
    model: llm.Model
        The language model to use.
    role: str
        The role of the model in the conversation.
    keep_history: bool
        Whether to remember previous rounds of the conversation. When `False`
        the language model will not keep any context. When set to `True`, the
        language model will remember all the previous rounds of the
        conversation.
    logger: LanguageModelLogger
        Where to log the conversation.
    options: dict[str, float | int]
        Model options (e.g., temperature, top-k, top-p, etc.).
    system: str | None, default is None
        The system prompt.
    """

    def __init__(
        self,
        model: llm.Model,
        role: str,
        keep_history: bool,
        logger: LanguageModelLogger,
        options: dict[str, float | int],
        system: str | None = None,
    ) -> None:
        self.model_id = model.model_id
        self.role = role
        self.keep_history = keep_history

        self._model = model
        self._logger = logger
        self._options = options
        self._system = system
        self._conversation: llm.models.Conversation
        self._responses: list[Response]

        self.clear()

    @property
    def responses(self) -> list[Response]:
        return self._responses

    def options(self) -> dict[str, float | int]:
        return self._options

    def conversation(self, system: str | None = None) -> LanguageModel:
        """Spawn a fresh conversation."""
        return Conversation(
            model=self._model,
            role=self.role,
            keep_history=True,
            logger=self._logger,
            options=self.options(),
            system=system,
        )

    def prompt(
        self,
        prompt: str,
        system: str | None = None,
        options: dict[str, float | int] | None = None,
    ) -> LanguageModelResponse:
        """Start or continue a conversation with an LLM."""
        if not self.keep_history:  # Start a fresh conversation
            self.clear()

        raw_response = _call_llm(
            self._conversation,
            prompt=prompt,
            system=system or self._system,
            options=options or self._options,
        )
        response = Response(raw_response, round=len(self._responses), role=self.role)

        self._responses.append(response)
        self._logger.response(response)

        return response

    def clear(self) -> None:
        """Clear the conversation history, if any."""
        self._conversation = self._model.conversation()
        self._responses = []


class LLM(Conversation):  # Conforms to LanguageModel
    """Protocol for interfacing with an LLM.

    This class uses Simon Willison's llm package to interact with a language
    model.

    Parameters
    ----------
    model_id: str
        The model identifier or alias. E.g., `mistralai/magistral-small`.
    role: str, default is "default"
        The role of the model in the conversation. For example: "attacker",
        "evaluator", "judge", etc.
    options: dict[str, float | int] | None, default is None
        Model-specific model options (e.g., temperature, top-k, etc.). These can
        be overridden on a per-prompt basis.
    keep_history: bool, default is False
        Whether the previous prompts should be sent along with any new prompt.
    system: str | None, default is None
        System prompt.
    logger: LanguageModelLogger | None, default is None
        Where to store the log of the conversation. By default, no log is kept.
    """

    def __init__(
        self,
        model_id: str,
        role: str = "default",
        options: dict[str, float | int] | None = None,
        keep_history: bool = False,
        system: str | None = None,
        logger: LanguageModelLogger | None = None,
    ) -> None:
        model = llm.get_model(model_id)

        super().__init__(
            model=model,
            role=role,
            keep_history=keep_history,
            logger=logger or NoOpLogger(),
            options=_validate_options(model=model, options=options or {}),
            system=system,
        )


class EmbeddingModel:  # Conforms to EmbeddingLanguageModel
    """An embedding model.

    Parameters
    ----------
    model_id: str
        The name or identifier of the embedding model to use.
    """

    def __init__(self, model_id: str) -> None:
        self.model = llm.get_embedding_model(model_id)

    def embed(self, text: str) -> ndarray:
        """Transform text into an embedding vector.

        Parameters
        ----------
        text: str
            The text to embed.

        Returns
        -------
        numpy.array
            The corresponding embedding vector.
        """
        return array(self.model.embed(text))


# vim: tw=80
