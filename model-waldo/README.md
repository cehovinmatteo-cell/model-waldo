# Model-Waldo

**Content-Aware MT/LLM Provider Selection Engine**

Model-Waldo analyzes real English source content and business constraints, then recommends the best MT provider or current LLM model using explainable scoring rules.

This is a hackathon prototype for the **GenAI in Localization Hackathon — Theme 2: Find the Right Model**.

## What it does

Model-Waldo is not a translation tool. It is a localization operations routing tool.

Workflow:

1. Upload or paste English source content.
2. The app extracts backend metadata: file type, word count, placeholders, tags, row/page counts.
3. OpenAI performs structured content analysis.
4. A local scoring engine ranks MT providers and specific LLM model variants.
5. The app presents a three-column decision dashboard: Recommended Model, Content Analysis, and Review / QA.
6. The full Model Ranking table is available in a collapsed section for transparency.

## MVP features

- English source content only
- Paste text input
- File upload for `.txt`, `.csv`, `.xlsx`, `.docx`, `.pdf`, `.json`, `.html`
- OpenAI structured content analysis
- Rule-based fallback mode if API key is missing or analysis fails
- Provider/model scoring engine
- Collapsible Model Ranking table
- Human review and QA recommendation for localized output
- Content Requirements checkboxes:
  - Glossary / terminology file required
  - Placeholder, tag, or format preservation required
  - Brand tone / transcreation required
- Separate sample test files in `/sample_data/`

## Privacy note

This MVP uses the OpenAI API for content analysis. Do not upload confidential, regulated, personal, or production-sensitive content unless you have the appropriate authorization and enterprise data controls in place.

## What is intentionally out of scope

- Non-English source content
- Live translation
- Live DeepL / Google / Microsoft / Amazon provider API integration
- BLEU / COMET / MQM scoring
- TMX support
- OCR or scanned PDF parsing
- Authentication
- PDF/Word report export
- Custom MT and Human + MT Hybrid as ranked provider answers

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add OpenAI API key

Either set an environment variable:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or copy the example Streamlit secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`.

### 3. Run the app

```bash
streamlit run app.py
```

## Recommended content-analysis model

Default: `gpt-5.4-mini`

Fallback: `gpt-5-mini`

Reason: this app needs structured classification, not deep reasoning. The model must be fast, multilingual, and reliable with JSON schema output.

## Provider/model database

The MVP ranks real MT providers and specific LLM model variants.

Traditional MT providers:

- DeepL
- Google Cloud Translation
- Microsoft Translator
- Amazon Translate

OpenAI:

- GPT-5.5
- GPT-5.4
- GPT-5.4 mini
- GPT-5.4 nano

Anthropic:

- Claude Opus 4.8
- Claude Sonnet 4.6
- Claude Haiku 4.5

Google Gemini:

- Gemini 2.5 Pro
- Gemini 2.5 Flash
- Gemini 2.5 Flash-Lite

The numeric scores are evidence-informed and demo-weighted. They should not be treated as official benchmark results. Cost remains an internal scoring factor when Business Priority is set to Cost, but the MVP does not display estimated dollar amounts.

## Sample datasets

Sample files are intentionally stored separately instead of being hard-coded into the app. This avoids the impression that demo results are pre-coded.

Use:

- `sample_data/ui_strings_en.csv`
- `sample_data/marketing_campaign_en.txt`
- `sample_data/legal_financial_en.txt`

Upload any of these files through the app to test the full routing pipeline.

## Evidence sources used for provider database context

Initial evidence layer:

- WMT24 General Machine Translation Task: https://aclanthology.org/2024.wmt-1.1/
- FLORES-200: https://ai.meta.com/tools/flores/
- Intento State of Translation Automation 2025: https://inten.to/the-state-of-translation-automation-2025/
- OpenAI model docs: https://developers.openai.com/api/docs/models/all
- OpenAI structured outputs docs: https://developers.openai.com/api/docs/guides/structured-outputs
- Anthropic API pricing/model docs: https://platform.claude.com/docs/
- Gemini API pricing/model docs: https://ai.google.dev/gemini-api/docs/pricing

## Important limitation

Public benchmarks are context-dependent. They do not perfectly predict provider/model quality for every language pair, customer domain, prompt, glossary setup, or review workflow.

Model-Waldo uses benchmark evidence as one layer, then combines it with:

- Actual English source content analysis
- Language pair complexity
- Business priority: None, Quality, or Cost
- Content type
- Content requirements
- Urgency
- Human review logic

## Suggested demo flow

1. Run the app.
2. Upload `sample_data/ui_strings_en.csv`.
3. Select `source_text` as the source column.
4. Source language is fixed to English.
5. Target language: German.
6. Business priority: None, Quality, or Cost.
7. Urgency: High.
8. Content Type: Auto-detect from source content or UI strings.
9. Optional Content Requirement: Placeholder, tag, or format preservation required.
10. Click **Run Model-Waldo**.
11. Show the decision dashboard, content analysis details, and collapsible Model Ranking table.
