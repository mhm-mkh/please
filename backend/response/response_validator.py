"""Basic response sanity checks before sending generated text to a user."""


class ResponseValidator:
    def validate(self, response: str | None, evidence: list[dict]) -> bool:
        if not isinstance(response, str) or not response.strip():
            return False
        known_ids = {str(item.get("id")) for item in evidence if item.get("id") is not None}
        import re
        citations = re.findall(r"\[source:([^\]]+)\]", response)
        if evidence and not citations:
            return False
        return all(citation in known_ids for citation in citations)
