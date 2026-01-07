import click
import httpx
import json
from llama_cpp import Llama
from llama_cpp.llama_chat_format import LlamaChatCompletionHandler
import llm
import logging
import pathlib

from click_default_group import DefaultGroup
from collections.abc import Callable, Iterable, Iterator
from pydantic import Field
from typing import cast, Union

logger = logging.getLogger(__name__)


def human_size(num_bytes: int) -> str:
    """Return a human readable byte size."""
    size = float(num_bytes)
    unit = "B"  # Makes pyright happy

    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0:
            break

        size /= 1024.0

    return f"{size:.2f} {unit}"


def _ensure_models_dir() -> pathlib.Path:
    directory = llm.user_dir() / "gguf" / "models"
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def _ensure_models_file() -> pathlib.Path:
    directory = llm.user_dir() / "gguf"
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / "models.json"

    if not filepath.exists():
        filepath.write_text("{}")

    return filepath


def _ensure_embed_models_file() -> pathlib.Path:
    directory = llm.user_dir() / "gguf"
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / "embed-models.json"

    if not filepath.exists():
        filepath.write_text("{}")

    return filepath


@llm.hookimpl
def register_models(register: Callable[..., None]) -> None:
    models_file = _ensure_models_file()
    models = json.loads(models_file.read_text())

    for model_id, info in models.items():
        model_path = info["path"]
        aliases = info.get("aliases", [])
        model_id = f"gguf/{model_id}"
        model = GgufChatModel(
            model_id,
            model_path,
            n_ctx=info.get("n_ctx", 0),
            n_gpu_layers=info.get("n_gpu_layers", 0),
        )
        register(model, aliases=aliases)


@llm.hookimpl
def register_embedding_models(register: Callable[..., None]) -> None:
    models_file = _ensure_embed_models_file()
    models = json.loads(models_file.read_text())

    for model_id, info in models.items():
        model_path = info["path"]
        aliases = info.get("aliases", [])
        model_id = f"gguf/{model_id}"
        model = GgufEmbeddingModel(model_id, model_path)
        register(model, aliases=aliases)


@llm.hookimpl
def register_commands(cli: DefaultGroup) -> None:
    @cli.group()
    def gguf() -> None:
        "Commands for working with GGUF models"

    @gguf.command()
    def models_file() -> None:
        "Display the path to the gguf/models.json file"
        directory = llm.user_dir() / "gguf"
        directory.mkdir(parents=True, exist_ok=True)
        models_file = directory / "models.json"
        click.echo(models_file)

    @gguf.command()
    def embed_models_file() -> None:
        "Display the path to the gguf/embed-models.json file"
        directory = llm.user_dir() / "gguf"
        directory.mkdir(parents=True, exist_ok=True)
        models_file = directory / "embed-models.json"
        click.echo(models_file)

    @gguf.command()
    def models_dir() -> None:
        "Display the path to the directory holding downloaded GGUF models"
        click.echo(_ensure_models_dir())

    @gguf.command()
    @click.argument("url")
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def download_model(url: str, aliases: list[str]) -> None:
        "Download and register a GGUF model from a URL"
        download_gguf_model(url, _ensure_models_file, aliases)

    @gguf.command()
    @click.argument("url")
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def download_embed_model(url: str, aliases: list[str]) -> None:
        "Download and register a GGUF embedding model from a URL"
        download_gguf_model(url, _ensure_embed_models_file, aliases)

    @gguf.command()
    @click.argument("model_id")
    @click.argument(
        "filepath", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
    )
    @click.option("n_ctx", "--n-ctx", type=int, default=0)
    @click.option("ngl", "--ngl", type=int, default=0)
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def register_model(
        model_id: str, filepath: str, n_ctx: int, aliases: tuple[str], ngl: int
    ) -> None:
        "Register a GGUF model that you have already downloaded with LLM"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        path = pathlib.Path(filepath)
        info = {
            "path": str(path.resolve()),
            "aliases": list(aliases),
            "n_gpu_layers": ngl,
        }

        if n_ctx:
            info["n_ctx"] = n_ctx

        models[model_id] = info
        models_file.write_text(json.dumps(models, indent=2))

    @gguf.command()
    @click.argument("model_id")
    @click.argument(
        "filepath", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
    )
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def register_embed_model(model_id: str, filepath: str, aliases: tuple[str]) -> None:
        "Register a GGUF embedding model that you have already downloaded"
        models_file = _ensure_embed_models_file()
        models = json.loads(models_file.read_text())
        path = pathlib.Path(filepath)
        info = {
            "path": str(path.resolve()),
            "aliases": list(aliases),
        }
        models[model_id] = info
        models_file.write_text(json.dumps(models, indent=2))

    @gguf.command()
    def models() -> None:
        "List registered GGUF models"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())

        for model, info in models.items():
            try:
                info["size"] = human_size(pathlib.Path(info["path"]).stat().st_size)
            except FileNotFoundError:
                info["size"] = None

        click.echo(json.dumps(models, indent=2))

    @gguf.command()
    def embed_models() -> None:
        "List registered GGUF embedding models"
        models_file = _ensure_embed_models_file()
        models = json.loads(models_file.read_text())

        for model, info in models.items():
            try:
                info["size"] = human_size(pathlib.Path(info["path"]).stat().st_size)
            except FileNotFoundError:
                info["size"] = None

        click.echo(json.dumps(models, indent=2))


class GgufChatModel(llm.Model):
    """Python wrapper for LLM models in GGUF format.

    Parameters
    ----------
    model_id: str
        The model identifier (e.g., "gguf/Qwen/Qwen3-1.7B-Q4_K_M").
    model_path: str
        The path where of the corresponding GGUF file.
    n_ctx: int, default is 0
        Context length. By default, this is read from the model.
    n_gpu_layers: int, default is 0
        Number of layers to offload to GPU (`-ngl` in llama.cpp). If -1, all
        layers are offloaded. The default is to load the model on the CPU.
    chat_handler: llama_cpp.llama_chat_format.LlamaChatCompletionHandler | None, default is None
        Optional chat handler to use when calling `Llama.create_chat_completion()`.
    """

    can_stream = True

    class Options(llm.Options):  # type: ignore
        max_tokens: int = Field(
            description="Maximum number of tokens to generate (default: 1000).",
            ge=0,
            default=1000,
        )
        temperature: float = Field(
            description="The model temperature (default: 0.7).",
            ge=0,
            default=0.7,
        )
        top_p: float = Field(
            description=(
                "Randomly sample at each generation step from the top "
                "most likely tokens whose probabilities add up to top_p."
                "(default: 0.8)."
            ),
            ge=0,
            le=1,
            default=0.8,
        )

    def __init__(
        self,
        model_id: str,
        model_path: str,
        n_ctx: int = 0,
        n_gpu_layers: int = 0,
        chat_handler: LlamaChatCompletionHandler | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.chat_handler = chat_handler
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._model: Llama | None = None  # Loaded lazily

        logger.debug("[llm-gguf] %s initialized.", self.model_id)

    def execute(
        self,
        prompt: llm.models.Prompt,
        stream: bool,
        response: llm.models.Response,
        conversation: llm.models.Conversation | None,
    ) -> Iterator[str]:
        if self._model is None:
            self._model = self._load_model()

            logger.debug("[llm-gguf] %s loaded", self.model_id)

        # These options are already merged with the class defaults
        options = dict(prompt.options)

        logger.debug("[llm-gguf] %s %s", self.model_id, options)

        messages = []
        current_system_prompt: str | None = None

        if conversation is not None:
            prev_responses = conversation.responses

            for prev_response in prev_responses:
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system_prompt
                ):
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                    current_system_prompt = prev_response.prompt.system

                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": cast(llm.models.Response, prev_response).text(),
                    }
                )

        if prompt.system and prompt.system != current_system_prompt:
            messages.append({"role": "system", "content": prompt.system})

        messages.append({"role": "user", "content": prompt.prompt})

        completion = self._model.create_chat_completion(
            messages=messages, stream=stream, **options
        )

        if stream:
            for chunk in completion:
                choice: dict = chunk["choices"][0]  # type: ignore
                delta_content = choice.get("delta", {}).get("content")

                if delta_content is not None:
                    yield delta_content
        else:
            yield completion["choices"][0]["message"]["content"]  # type: ignore

    def _load_model(self) -> Llama:
        return Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
            n_ctx=self.n_ctx,
            chat_handler=self.chat_handler,
        )


class GgufEmbeddingModel(llm.models.EmbeddingModel):
    def __init__(self, model_id: str, model_path: str) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self._model: Llama | None = None

    def embed_batch(self, items: Iterable[Union[str, bytes]]) -> Iterator[list[float]]:
        if self._model is None:
            self._model = Llama(
                model_path=self.model_path, embedding=True, verbose=False
            )

        # `results` has type llama_types.CreateEmbeddingResponse, which is
        # a (typed) dictionary with a "data" field. The "data" field is a list
        # of llama_types.Embedding, which is itself a (typed) dictionary with an
        # "embedding" field of type Union[list[float], list[list[float]].
        results = self._model.create_embedding([str(item) for item in items])

        for result in results["data"]:
            yield cast(list[float], result["embedding"])


def download_gguf_model(
    url: str, models_file_func: Callable, aliases: list[str]
) -> None:
    """Download a GGUF model and register it in the specified models file"""
    with httpx.stream("GET", url, follow_redirects=True) as response:
        total_size = response.headers.get("content-length")

        filename = url.split("/")[-1]
        download_path = _ensure_models_dir() / filename

        if download_path.exists():
            raise click.ClickException(f"File already exists at {download_path}")

        with open(download_path, "wb") as fp:
            if total_size is not None:
                total_size = int(total_size)

                with click.progressbar(
                    length=total_size,
                    label="Downloading {}".format(human_size(total_size)),
                ) as bar:
                    for data in response.iter_bytes(1024):
                        fp.write(data)
                        bar.update(len(data))
            else:
                for data in response.iter_bytes(1024):
                    fp.write(data)

        click.echo(f"Downloaded model to {download_path}", err=True)
        models_file = models_file_func()
        models = json.loads(models_file.read_text())
        model_id = download_path.stem
        info = {
            "path": str(download_path.resolve()),
            "aliases": aliases,
            "n_gpu_layers": 0,
            "n_ctx": 0,
        }
        models[model_id] = info
        models_file.write_text(json.dumps(models, indent=2))
