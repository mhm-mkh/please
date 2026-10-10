"""Translate Streamlit values to the shared request/response contracts."""

from backend.request_models import AssistantResponse, MessageRequest


class StreamlitAdapter:
    def to_request(self, text: str, conversation_id: str) -> MessageRequest:
        return MessageRequest(text=text, conversation_id=conversation_id, platform="streamlit")

    @staticmethod
    def to_message(response: AssistantResponse) -> dict:
        return {"role": "assistant", "content": response.answer,
                "results": response.results, "links": response.links}
