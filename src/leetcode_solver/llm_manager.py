from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from langchain.output_parsers import PydanticOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, create_model

T = TypeVar("T", bound=BaseModel)


class OutputMode:
    """Enum-like class for output modes"""

    RAW = "raw"
    JSON = "json"
    STRUCTURED = "structured"


class LLMManager:
    """
    A manager class that provides a unified interface for different LLM providers
    with support for raw output, JSON mode, and structured output using Pydantic
    """

    def __init__(self):
        self.models: Dict[str, BaseChatModel] = {}

    def add_model(self, name: str, model: BaseChatModel) -> None:
        """Register a new model with the manager"""
        self.models[name] = model

    def setup_default_models(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ) -> None:
        """Setup default configurations for common LLM providers"""
        if openai_api_key:
            # OpenAI models with JSON mode support
            self.add_model(
                "gpt-4o",
                ChatOpenAI(
                    model_name="gpt-4o", api_key=openai_api_key, temperature=0.7
                ),
            )
            self.add_model(
                "gpt-4o-json",
                ChatOpenAI(
                    model_name="gpt-4o",
                    api_key=openai_api_key,
                    temperature=0.7,
                    # response_format={"type": "json_object"}
                ),
            )

        if anthropic_api_key:
            # Anthropic models
            self.add_model(
                "claude-3-5",
                ChatAnthropic(
                    model="claude-3-5-sonnet-latest",
                    api_key=anthropic_api_key,
                    temperature=0.7,
                ),
            )

        if google_api_key:
            # Gemini model
            self.add_model(
                "gemini-flash",
                ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    google_api_key=google_api_key,
                    temperature=0.7,
                ),
            )

    def _get_output_parser(
        self, output_mode: str, pydantic_schema: Optional[Type[BaseModel]] = None
    ):
        """Get appropriate output parser based on mode and schema"""
        if output_mode == OutputMode.STRUCTURED and pydantic_schema:
            return PydanticOutputParser(pydantic_object=pydantic_schema)
        elif output_mode == OutputMode.JSON:
            return JsonOutputParser()
        return StrOutputParser()

    def _prepare_format_message(
        self, output_mode: str, pydantic_schema: Optional[Type[BaseModel]] = None
    ):
        """Prepare messages with appropriate formatting instructions"""
        messages = []

        if output_mode == OutputMode.STRUCTURED and pydantic_schema:
            parser = PydanticOutputParser(pydantic_object=pydantic_schema)
            # Add formatting instructions as system message
            messages.append(
                {
                    "role": "system",
                    "content": f"""
                You must respond in the following format: {parser.get_format_instructions()}
            """,
                }
            )
        elif output_mode == OutputMode.JSON:
            messages.append(
                {
                    "role": "system",
                    "content": "Respond with a valid JSON object only. Ensure the response is properly formatted JSON.",
                }
            )

        return messages

    def generate(
        self,
        prompts: List[dict],
        model_name: str,
        output_mode: str = OutputMode.RAW,
        pydantic_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Union[str, dict, BaseModel]:
        """
        Synchronous generate function with support for different output modes
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        format_prompt = self._prepare_format_message(output_mode, pydantic_schema)
        parser = self._get_output_parser(output_mode, pydantic_schema)

        # Use JSON-specific model if available
        if output_mode == OutputMode.JSON and f"{model_name}-json" in self.models:
            model = self.models[f"{model_name}-json"]

        response = model.invoke(format_prompt + prompts, **kwargs)

        if parser:
            try:
                return parser.parse(response.content)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse response: {e}\nResponse: {response.content}"
                )

        return response.content


from typing import List

# Example usage:
from pydantic import BaseModel, Field


class Scientist(BaseModel):
    name: str = Field(description="The scientist's full name")
    field: str = Field(description="The primary field of study")
    contributions: List[str] = Field(
        description="List of major contributions to science"
    )
    year_of_birth: int = Field(description="Year the scientist was born")


class ScientistList(BaseModel):
    scientists: List[Scientist] = Field(
        description="List of scientists and their details"
    )


async def example_usage():
    llm_manager = LLMManager()
    llm_manager.setup_default_models(
        openai_api_key="your-openai-key", anthropic_api_key="your-anthropic-key"
    )

    # 1. Raw text output
    raw_response = await llm_manager.agenerate(
        prompt="List three famous scientists.",
        model_name="gpt-4",
        output_mode=OutputMode.RAW,
    )
    print("Raw response:", raw_response)

    # 2. JSON output (less structured)
    json_response = await llm_manager.agenerate(
        prompt="List three famous scientists.",
        model_name="gpt-4",
        output_mode=OutputMode.JSON,
    )
    print("JSON response:", json_response)

    # 3. Structured output with Pydantic schema
    structured_response = await llm_manager.agenerate(
        prompt="List three famous scientists who made significant contributions to physics.",
        model_name="gpt-4",
        output_mode=OutputMode.STRUCTURED,
        pydantic_schema=ScientistList,
    )
    print("Structured response:", structured_response)


# Synchronous usage example
def sync_example():
    llm_manager = LLMManager()
    llm_manager.setup_default_models(openai_api_key="your-openai-key")

    # Using structured output with Pydantic schema
    response = llm_manager.generate(
        prompt="List two famous scientists who contributed to quantum mechanics.",
        model_name="gpt-4",
        output_mode=OutputMode.STRUCTURED,
        pydantic_schema=ScientistList,
    )

    # Access structured data with type hints
    for scientist in response.scientists:
        print(f"{scientist.name} ({scientist.year_of_birth})")
        print(f"Field: {scientist.field}")
        print(f"Contributions: {', '.join(scientist.contributions)}")
