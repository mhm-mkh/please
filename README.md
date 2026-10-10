# Modular backend application

This is the new modular project location. The earlier `streamlit-free-precomputed` folder is unchanged. The former Python modules are preserved in `legacy_reference/`; the new Streamlit path uses the modular files.

## Run locally or deploy

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, upload this folder to GitHub and set **Main file path** to `app.py`. Keep these files committed together:

- `data/knowledge/canada.jsonl`
- `data/knowledge/canada.faiss`
- `data/knowledge/canada.index.json`
- `data/pathways/pathways.json`

The supplied index matches the included corpus: 207 chunks, model `codefuse-ai/F2LLM-v2-0.6B`, dimension 1024, fingerprint `dea1c79ff18cc8548a5da4b6532ac30c27a3e7300f66cc55db0a86187dca4621`. Startup verifies the corpus fingerprint, vector count, and model dimension. Rebuild the index after changing corpus text/order or model with `python scripts/build_index.py`.

The app loads the embedding model at startup to encode questions. The host needs network access to download it on first startup unless it is already cached. Startup reports and stops on missing or mismatched corpus/index files.

## LLM responses

The app works without LLM credentials: it returns a safe template response and displays retrieved evidence in expanders. To enable generated answers, configure Streamlit secrets or environment variables:

```toml
LLM_API_KEY = "your-provider-key"
LLM_MODEL = "your-chat-model"
LLM_BASE_URL = "https://api.openai.com/v1"
```

The client calls an OpenAI-compatible `/chat/completions` endpoint. With no key/model it skips generation. If a configured request fails or its answer fails citation validation, the app uses the template fallback and still shows retrieved evidence. Never commit secrets to GitHub.

## Request flow

`platform adapter → MessageRequest → message_handler → applicable paths → combined results → response creator → AssistantResponse → originating platform`.

The Streamlit adapter is implemented. `platform_2`, `platform_3`, and `platform_4` remain placeholders because their platform names have not been specified. Admin helpers are separate from public chat and need access control before deployment.

## Consultation requests and email delivery

The app offers an optional consultation request after several profile facts have been collected. It requires a name, phone number, and explicit consent. On consent, it creates a Persian PDF in memory and sends it to exactly one consultant selected from a private round-robin list. Recipient addresses are read only from Streamlit secrets and are never shown in the user interface. Email is not sent to multiple recipients or copied to other consultants.

Configure these values in Streamlit Cloud **Manage app → Settings → Secrets**. Keep them out of GitHub:

```toml
CONSULTANT_EMAILS = ["your-private-test-recipient@example.com"]
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "the-sending-account@example.com"
SMTP_PASSWORD = "the-sending-account-app-password"
SMTP_FROM_EMAIL = "the-sending-account@example.com"
CONSULTATION_FINGERPRINT_SECRET = "replace-with-a-random-secret-at-least-32-characters-long"
```

`CONSULTANT_EMAILS` accepts a list, so later you can add consultant addresses there. Only one list entry is selected for each new phone number. Configure the test recipient in Streamlit Secrets; do not add it to source code or commit it.

The app stores a keyed phone fingerprint, assigned recipient, request ID, and delivery state in `data/private/consultation-requests.sqlite3` to prevent the same phone from being routed again. It does not store the PDF or profile in that database. This local SQLite file is suitable for a controlled test, but Streamlit Cloud's app filesystem is not a durable shared database. Before production or multiple app instances, move this small repository to a persistent shared database; otherwise a restart can lose duplicate-prevention history. Failed sends remain blocked to prevent accidental duplicate routing and need administrative review.

## Limitations

- The available pathway definitions currently cover Canada only.
- Pathway scores are preliminary profile-data coverage and preference ranking, not legal eligibility. The pathway data still needs verified cost, timeline, and work-permit details before those can be compared.
- Profile and conversation state remain in the active Streamlit session; there is no long-term profile database.
- Without an LLM, the response is a safe template plus retrieved source passages.
