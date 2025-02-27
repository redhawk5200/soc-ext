from datetime import datetime
from datetime import timezone
from enum import StrEnum
import logging
import os
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import backoff
import openai
from openai import AsyncOpenAI
from openai import AzureOpenAI
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionContentPartParam
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionSystemMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_chunk import Choice as StreamChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel
import tiktoken
from typing_extensions import TypeVar

logger = logging.getLogger("cape")


class LLMError(Exception):
    """
    Exception raised for errors with the LLM.
    """

    def __init__(self, e: openai.OpenAIError) -> None:
        self.data = {}
        if isinstance(e, openai.APIConnectionError):
            self.message = f"LLM request failed to connect: {e}"
        elif isinstance(e, openai.BadRequestError):
            self.message = e.message
            self.data = {
                "code": e.code,
            }
        elif isinstance(e, openai.RateLimitError):
            self.message = f"LLM request was rate limited, please try again: {e}"
        elif isinstance(e, openai.APIError):
            self.message = f"LLM returned an API error: {e}"
        else:
            self.message = f"LLM returned error: {e}"

        logger.error(self.message)

    def __str__(self) -> str:
        return str(self.message)

    def __repr__(self) -> str:
        return repr(self.message)


class Role(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: Union[str, Iterable[ChatCompletionContentPartParam]]


class CompletionChunk(BaseModel):
    message: str


class Completion:
    def __init__(self, iterator: Iterator[ChatCompletionChunk] | ChatCompletion):
        self._iterator = iterator

    def message(self) -> str:
        if isinstance(self._iterator, ChatCompletion):
            if len(self._iterator.choices) == 0:
                return ""

            return self._iterator.choices[0].message.content or ""
        else:
            resp = ""
            for chunk in self._iterator:
                if len(chunk.choices) == 0:
                    continue

                resp += chunk.choices[0].delta.content or ""

            return resp

    def __iter__(self) -> Iterator[CompletionChunk]:
        if isinstance(self._iterator, ChatCompletion):
            if len(self._iterator.choices) == 0:
                iter = create_chat_completions_iterator("")
            else:
                iter = create_chat_completions_iterator(self._iterator.choices[0].message.content or "")

            return map(lambda x: CompletionChunk(message=x.choices[0].delta.content or ""), iter)

        def f(x: ChatCompletionChunk) -> CompletionChunk:
            if len(x.choices) == 0:
                return CompletionChunk(message="")

            return CompletionChunk(message=x.choices[0].delta.content or "")

        return map(f, self._iterator)

    def json_iter(self) -> Iterator[str]:
        if isinstance(self._iterator, ChatCompletion):
            if len(self._iterator.choices) == 0:
                iter = create_chat_completions_iterator("")
            else:
                iter = create_chat_completions_iterator(self._iterator.choices[0].message.content or "")
            return map(lambda x: x.model_dump_json(), iter)

        return map(lambda x: x.model_dump_json(), self._iterator)


class AsyncCompletion:
    def __init__(self, iterator: AsyncIterator[ChatCompletionChunk]):
        self._iterator = iterator

    async def message(self) -> str:
        resp = ""
        async for chunk in self._iterator:
            if len(chunk.choices) == 0:
                continue

            resp += chunk.choices[0].delta.content or ""

        return resp

    async def __iter__(self) -> Iterator[CompletionChunk]:
        i = []
        async for x in self._iterator:
            if len(x.choices) == 0:
                continue

            i.append(CompletionChunk(message=x.choices[0].delta.content or ""))
        return i.__iter__()

    async def json_iter(self) -> Iterator[str]:
        i = []
        async for x in self._iterator:
            i.append(x.model_dump_json())
        return i.__iter__()


# because this code is _very_ temporary, playing fast and loose w/ types
# FIXME: some llama3 models haven't been patched with the updated generation_config.yaml from the upstream model repo
# particularly, we're using a gptq quant that needs to be updated.
# would be better to update or fork the model repo with the correct changes so we can remove this patch for good.
def _llama3_patch_create(f: Any) -> Any:
    def _patched(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        if len(args) != 0:
            raise ValueError("kwargs only for client.chat.completions.create")

        model: Optional[str] = kwargs.get("model")  # type: ignore[assignment]
        if model is not None:
            if "llama-3-8b-instruct-gptq" in model.lower():
                kwargs["extra_body"] = {"stop_token_ids": [128001, 128009]}
        return f(**kwargs)

    return _patched


def get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    if api_key is None or api_key == "":
        api_key = os.environ.get("OPENAI_API_KEY", "sk-123")

    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1/")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    if "openai.azure.com" in base_url:
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=base_url,
        )

    # FIXME: some llama3 models haven't been patched with the updated generation_config.yaml from the upstream model repo
    # particularly, we're using a gptq quant that needs to be updated.
    # would be better to update or fork the model repo with the correct changes so we can remove this patch for good.
    client.chat.completions.create = _llama3_patch_create(client.chat.completions.create)  # type: ignore[method-assign]
    return client


def get_openai_async_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> AsyncOpenAI:
    if api_key is None or api_key == "":
        api_key = os.environ.get("OPENAI_API_KEY", "sk-123")

    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1/")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # FIXME: some llama3 models haven't been patched with the updated generation_config.yaml from the upstream model repo
    # particularly, we're using a gptq quant that needs to be updated.
    # would be better to update or fork the model repo with the correct changes so we can remove this patch for good.
    client.chat.completions.create = _llama3_patch_create(client.chat.completions.create)  # type: ignore[method-assign]
    return client


ResponseFormatT = TypeVar(
    "ResponseFormatT",
    # if it isn't given then we don't do any parsing
    default=None,
)


def parse(
    client: OpenAI,
    model: str,
    messages: Iterable[ChatCompletionMessageParam],
    response_format: type[ResponseFormatT],
) -> ResponseFormatT:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
    )

    message = completion.choices[0].message
    if message.parsed:
        return message.parsed

    raise Exception(message.refusal)


@backoff.on_exception(backoff.expo, (LLMError, openai.OpenAIError), max_tries=3)
def chat_completions(
    client: OpenAI,
    model: str,
    messages: List[Message],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = True,
    response_format: Optional[Dict[str, str]] = None,
) -> Completion:
    openai_messages: List[ChatCompletionMessageParam] = []

    for m in messages:
        match m.role:
            case Role.system:
                if not isinstance(m.content, str):
                    raise ValueError("expected content to be string to system msg")
                openai_messages.append(ChatCompletionSystemMessageParam(role="system", content=m.content))
            case Role.user:
                openai_messages.append(ChatCompletionUserMessageParam(role="user", content=m.content))
            case Role.assistant:
                if not isinstance(m.content, str):
                    raise ValueError("expected content to be string to assistant msg")
                openai_messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=m.content))

    request_options = {
        "messages": openai_messages,
        "temperature": temperature,
        "stream": stream,
        "model": model,
    }
    if max_tokens is not None:
        request_options["max_tokens"] = max_tokens

    if response_format is not None:
        request_options["response_format"] = response_format

    # FIXME: some llama3 models haven't been patched with the updated generation_config.yaml from the upstream model repo
    # particularly, we're using a gptq quant that needs to be updated.
    # would be better to update or fork the model repo with the correct changes so we can remove this patch for good.
    if "llama-3-8b-instruct-gptq" in model.lower():
        request_options["extra_body"] = {"stop_token_ids": [128001, 128009]}

    try:
        gen = client.chat.completions.create(**request_options)  # type: ignore[call-overload]
    except openai.OpenAIError as e:
        raise LLMError(e)

    return Completion(gen)


async def async_chat_completions(
    client: AsyncOpenAI,
    model: str,
    messages: List[Message],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
) -> AsyncCompletion:
    openai_messages: List[ChatCompletionMessageParam] = []

    for m in messages:
        match m.role:
            case Role.system:
                if not isinstance(m.content, str):
                    raise ValueError("expected content to be string to system msg")

                openai_messages.append(ChatCompletionSystemMessageParam(role="system", content=m.content))
            case Role.user:
                openai_messages.append(ChatCompletionUserMessageParam(role="user", content=m.content))
            case Role.assistant:
                if not isinstance(m.content, str):
                    raise ValueError("expected content to be string to system msg")

                openai_messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=m.content))
    request_options = {
        "messages": openai_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "model": model,
        "extra_headers": extra_headers,
    }

    # FIXME: issue with vLLM ignoring generation_config.json of underlying llama model
    # remove this when we bump minimum vLLM version to 0.4.1, pending their release
    if "llama-3-8b-instruct-gptq" in model.lower():
        request_options["extra_body"] = {"stop_token_ids": [128001, 128009]}

    try:
        gen = await client.chat.completions.create(**request_options)  # type: ignore[call-overload]
    except openai.OpenAIError as e:
        raise LLMError(e)

    return AsyncCompletion(gen)


def create_chat_completions_iterator(response: str) -> Iterator[ChatCompletionChunk]:
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = [enc.decode([t]) for t in enc.encode(response)]
    for t in tokens:
        yield ChatCompletionChunk(
            choices=[
                StreamChoice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content=t,
                    ),
                )
            ],
            created=int(datetime.now(timezone.utc).timestamp()),
            object="chat.completion.chunk",
            model="gpt-4",
            id="foo",
        )


def create_chat_completion(response: str) -> Completion:
    return Completion(create_chat_completions_iterator(response))


