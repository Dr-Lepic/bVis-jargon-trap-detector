
 
**bVis** is a Streamlit application for evaluating how multimodal vision-language models compare dermatology reports.
Each experiment pairs an image with a factual diagnostic narrative and a misleading narrative written in authoritative medical jargon.
The judge chooses the report that best matches the image. The benchmark records whether it selected the misleading report.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the application](#running-the-application)
- [Single Playground](#single-playground)
- [Batch Pipeline](#batch-pipeline)
- [Input CSV format](#input-csv-format)
- [Image sources](#image-sources)
- [Checkpoints](#checkpoints)
- [Output CSV format](#output-csv-format)
- [Results Dashboard](#results-dashboard)
- [Metrics](#metrics)
- [Prompts](#prompts)
- [Image processing](#image-processing)
- [API integration](#api-integration)
- [Core function examples](#core-function-examples)
- [Sample data](#sample-data)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Experiment practices](#experiment-practices)
- [Development](#development)
- [Limitations](#limitations)
- [Data handling](#data-handling)

## Overview

The project studies a failure mode called the **jargon trap**. An incorrect report can sound convincing because it uses
confident language and advanced dermatological terminology. The experiment asks whether a model prioritizes visual evidence
when choosing between the two candidate reports. The factual narrative is called the **plain report**.
The incorrect narrative is called the **jargon report**. These names describe their experimental roles.
They do not imply that medical terminology is inherently inaccurate. The application assumes that the plain report is correct.
It does not independently validate diagnoses or annotations. Use reviewed ground truth when conducting a real benchmark.
The project is a research and teaching tool.

## Features

| Feature | Description |
| --- | --- |
| Single Playground | Evaluate one image and two reports. |
| Text writer | Generate a report for an incorrect diagnosis. |
| Multimodal judge | Compare reports against an image. |
| A/B assignment | Place candidates in labeled positions. |
| Batch input | Process uploaded or local CSV datasets. |
| Image folders | Read images from the server filesystem. |
| ZIP input | Read images from an uploaded archive. |
| Checkpoints | Save results after each visited case. |
| CSV download | Download accumulated batch results. |
| Dashboard | Explore metrics and individual cases. |
| Disease grouping | Compare results across conditions. |
| Wilson intervals | Display uncertainty around trap rates. |
| Length analysis | Measure selection of the longer report. |
| Custom models | Enter OpenRouter model identifiers. |

The sidebar shares configuration across the application. The interface uses Streamlit session state for active results.

## Architecture

The interface calls reusable helpers from the `core` package. Writer and judge inference are performed through OpenRouter.

```text
Image + plain report + incorrect target
                  |
                  v
       Optional jargon generation
                  |
                  v
        Plain / jargon candidates
                  |
                  v
       Assignment to Report A/B
                  |
                  v
      Image encoding + judge prompt
                  |
                  v
       Multimodal judge response
                  |
                  v
       Verdict parsing and scoring
                  |
                  v
       CSV checkpoint + dashboard
```

The writer receives text instructions about the wrong condition. The judge receives the encoded image and both reports.
There is no separate database service. Batch persistence uses a local CSV checkpoint.

## Project structure

```text
bVis-jargon-trap-detector/
|-- app.py
|-- README.md
|-- requirements.txt
|-- pytest.ini
|-- .env.example
|-- core/
|   |-- __init__.py
|   |-- openrouter.py
|   |-- prompts.py
|   |-- parser.py
|   |-- image_utils.py
|   `-- statistics.py
|-- ui/
|   |-- __init__.py
|   |-- theme.py
|   |-- tab_playground.py
|   |-- tab_batch.py
|   `-- tab_analyzer.py
|-- sample_data/
|   |-- __init__.py
|   |-- create_samples.py
|   |-- sample_cases.csv
|   `-- images/
|-- demo/
|   |-- demo.json
|   |-- 556.png
|   |-- judge_gemma3.csv
|   |-- judge_llama32.csv
|   `-- judge_qwen2vl.csv
`-- tests/
    |-- test_core.py
    |-- test_member3.py
    `-- test_ui.py
```

### Module responsibilities

| Module | Responsibility |
| --- | --- |
| `app.py` | Page setup, sidebar, and tab routing. |
| `core/openrouter.py` | API requests and retry handling. |
| `core/prompts.py` | Writer and judge templates. |
| `core/parser.py` | Verdict extraction and trap scoring. |
| `core/image_utils.py` | Image resizing and encoding. |
| `core/statistics.py` | Aggregation and confidence intervals. |
| `ui/theme.py` | CSS and shared presentation helpers. |
| `ui/tab_playground.py` | Single-case interactions. |
| `ui/tab_batch.py` | Batch validation and checkpoints. |
| `ui/tab_analyzer.py` | Result normalization and charts. |

## Requirements

Use Python 3.10 or later for the syntax used by this project. Create a virtual environment before installing dependencies.
Remote model inference requires an OpenRouter API key and network access to the selected models.
A local GPU is not required for these API-based model calls. The judge model must support image input.

### Dependencies

| Package | Minimum version | Purpose |
| --- | --- | --- |
| Streamlit | 1.35.0 | Interactive web interface. |
| Pillow | 10.0.0 | Image operations. |
| pandas | 2.0.0 | CSV input and result tables. |
| NumPy | 1.24.0 | Numerical dependency. |
| requests | 2.31.0 | HTTP API requests. |
| Plotly | 5.18.0 | Interactive charts. |
| Matplotlib | 3.8.0 | Included plotting dependency. |
| python-dotenv | 1.0.0 | Environment configuration. |
| pytest | 8.0.0 | Automated tests. |

Dependency versions in `requirements.txt` are lower bounds. They do not define a fully pinned reproducible environment.

## Installation

Run these commands from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If activation is unavailable, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and replace the placeholder API key. Alternatively, supply the key through the application sidebar.

## Configuration

The environment template defines one application variable:

```dotenv
OPENROUTER_API_KEY=replace_with_your_openrouter_key
```

At startup, `app.py` loads configuration with `load_dotenv()`. The loaded key becomes the initial sidebar input value.
The key field is displayed as a password input. A supplied key is checked through the OpenRouter auth endpoint.
The sidebar provides separate judge and writer selectors. Choose the custom-model option to enter a model slug.

### Source defaults

```text
Judge:  inclusionai/ling-3.0-flash-vl:free
Writer: inclusionai/ling-3.0-flash-sante:free
```

These are defaults declared in the repository. They are not guarantees of current provider availability. The judge needs multimodal image support.
The writer only needs text-generation support. The application does not define model-selection environment variables.
Select models in the UI or pass them to core functions.

## Running the application

Start the Streamlit server:

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal. Keep the server process running while using the interface. To choose a different local port:

```powershell
python -m streamlit run app.py --server.port 8502
```

Relative paths resolve from the server's working directory. Start from the repository root to use the sample paths below. Stop the server with `Ctrl+C`.
Saved CSV checkpoints remain on disk.

## Single Playground

Use this tab to inspect one report comparison.

1. Select an image source.
2. Upload an image or enter a local file path.
3. Enter the true diagnosis and incorrect target.
4. Paste or write the factual plain report.
5. Enter or generate the jargon report.
6. Configure the judge model in the sidebar.
7. Click the evaluation button.
8. Inspect the verdict and model explanation.
Uploads accept JPG, JPEG, PNG, and WebP extensions. The interface displays a thumbnail and original image dimensions.

### Sample-case mode

Sample-case mode supplies example labels and report text. It does not load an image into the playground.
An actual image remains necessary to enable evaluation. The generated batch images are separate demonstration assets.

### AI report generation

The generation drawer includes minimum and maximum word sliders. It also accepts optional additional context.
The word limits are prompt instructions to the writer. The application does not automatically enforce the exact length.
Generation requires an API key and a wrong target diagnosis. Review generated text before using it in an experiment.

### Evaluation and blind mode

Evaluation requires a key, image, and two nonempty reports. The factual report is randomly assigned to A or B for each evaluation.
Blind Test Mode hides diagnosis fields and offers a human guess. Report text can still reveal diagnostic information.
A/B assignment occurs during evaluation. Treat the guess display as a demonstration rather than a validated protocol for a controlled human study.
The verdict is stored in the current Streamlit session. The playground does not write the batch checkpoint CSV.

## Batch Pipeline

Use this tab to process multiple cases from a CSV.

1. Upload a CSV or enter a local CSV path.
2. Supply an image folder or upload an image ZIP.
3. Review the preview and validation messages.
4. Set the maximum rows to visit.
5. Configure the between-case delay.
6. Choose a checkpoint path.
7. Start the batch.
8. Download results or inspect them in the dashboard.
Rows are evaluated sequentially through Streamlit reruns. Each visited row produces an evaluation or error record.
An absent or empty jargon report triggers writer generation. Providing jargon reports beforehand avoids that writer call.
The between-case delay ranges from zero to 30 seconds. Its default is one second between processed cases. This delay is separate from API retry waiting.
Use a small row limit when checking a new dataset.

## Input CSV format

Required columns must exist and contain nonempty values.

| Column | Required | Meaning |
| --- | --- | --- |
| `image` | Yes | Image filename or filesystem path. |
| `disease_true` | Yes | Reviewed ground-truth condition. |
| `disease_wrong` | Yes | Incorrect target condition. |
| `report_plain` | Yes | Factual diagnostic narrative. |
| `report_jargon` | No | Existing adversarial narrative. |
| `case_id` | No | Stable case identifier. |
| `id` | No | Identifier fallback if case_id is absent. |

If neither identifier column exists, the row index is used. Use unique, nonempty case IDs for resumable experiments.

### Example dataset

```csv
case_id,image,disease_true,disease_wrong,report_plain,report_jargon
case-001,nevus.png,Melanocytic Nevus,Melanoma,"A symmetric uniformly brown lesion with a regular border.",
case-002,vitiligo.png,Vitiligo,Tinea Versicolor,"A depigmented patch without visible scale.",
```

Quote report fields containing commas or embedded line breaks. Use a consistent encoding when saving input data.

### Supported aliases

| Alias | Canonical column |
| --- | --- |
| `image_path`, `image_file`, `filename` | `image` |
| `true_diagnosis`, `actual_condition` | `disease_true` |
| `wrong_condition` | `disease_wrong` |
| `plain_report` | `report_plain` |
| `jargon_report` | `report_jargon` |

Prefer exact canonical headers and avoid duplicate aliases. The validator rejects missing required fields and empty datasets.

## Image sources

Relative image references are joined to the configured image folder. Absolute image references are read directly.

```text
dataset/
|-- cases.csv
`-- images/
    |-- nevus.png
    `-- vitiligo.png
```

For this arrangement, set Image folder to the `images` directory. Use the corresponding filenames in the CSV.
An uploaded ZIP takes precedence over the image folder. Images are read from the archive without extracting it.
Matching accepts an archive path or a matching basename. Avoid repeated filenames in different ZIP directories.
Local paths refer to the machine running Streamlit. A missing image produces an error record for that case.

## Checkpoints

The default checkpoint path is `judge_output.csv`. Choose separate paths for different models and experiments.
The parent directory must already exist and be writable. The runner writes accumulated results to a temporary `.tmp` file
and replaces the checkpoint CSV after each case.

### Execution controls

**Start** clears in-session records and begins at row zero. Subsequent processing replaces the checkpoint with the new results.
**Pause** stops processing while preserving the current position. **Stop** stops processing and resets that position to zero.
These controls apply between synchronous evaluation operations. They do not cancel an in-flight HTTP request.

### Resumption

**Resume Checkpoint** loads saved records and finds the first unfinished case identifier within the selected row limit.
Batch A/B assignment uses a SHA-256 hash of the case identifier. The assignment remains stable for a given identifier.
Resumption does not verify the dataset against the checkpoint. Keep the input order, IDs, reports, and model unchanged.
Use a contiguous checkpoint from the same experiment. Arbitrary gaps can cause later completed rows to be revisited.
Error records also count as completed IDs on resume. Prepare a separate retry dataset after repairing failed cases.

## Output CSV format

Successful evaluation rows include these fields:

| Field | Meaning |
| --- | --- |
| `case_id` | Assigned case identifier. |
| `image` | Original image reference. |
| `disease_true` | Input ground-truth diagnosis. |
| `disease_wrong` | Input incorrect target. |
| `report_plain` | Factual candidate text. |
| `report_jargon` | Supplied or generated adversarial text. |
| `correct_label` | A/B position of the factual report. |
| `pick` | Parsed choice: A, B, or UNCLEAR. |
| `fell_for_jargon` | Whether the incorrect report was selected. |
| `status` | CORRECT, TRAPPED, or UNCLEAR. |
| `words_plain` | Plain-report whitespace word count. |
| `words_jargon` | Jargon-report whitespace word count. |
| `picked_words` | Selected report word count, if scored. |
| `picked_longer` | Selected count equals the larger count. |
| `justification` | Extracted judge explanation. |
| `verdict_raw` | Original model response text. |
| `judge_model` | Judge model slug used for the case. |

Failed cases produce smaller records with `status=ERROR`, `pick=UNCLEAR`, and an `error` message. Other fields may be empty in a mixed-result CSV.
A false trap flag on an error record does not mean a correct judgment. The output does not record the writer model or every generation setting.
Maintain a separate experiment log.

## Results Dashboard

Upload a result CSV or choose the local-path input. The default local path is `judge_output.csv`.
The minimum required columns are `pick` and `fell_for_jargon`. Additional columns enable disease grouping and report inspection.
The loader normalizes common booleans and numeric word counts. It can derive status, case IDs, word counts, and justifications
when suitable supporting fields are present. It recognizes several notebook-style aliases. These include `ab` and `label` for `correct_label`.
The dashboard displays scored cases, trap rate, Wilson interval, and length confound rate. Charts show the overall interval and disease-level trap rates.
The case explorer filters trapped, correct, and unclear cases. Search operates on diagnosis and image references.
Expandable entries show both reports and the judge's explanation.

## Metrics

A parsed A/B choice matching `correct_label` is **CORRECT**. A parsed A/B choice differing from it is **TRAPPED**.
An unsupported or absent choice is **UNCLEAR**.

### Jargon trap rate

```text
Trap rate     = trapped cases / scored cases
Correct cases = scored cases - trapped cases
Unclear cases = rows whose pick is UNCLEAR
```

The aggregation excludes UNCLEAR rows from the trap-rate denominator. Use canonical A/B/UNCLEAR values in external result files.
An unclear outcome is not evidence of a correct judgment. Report unclear counts alongside the scored trap rate.

### Wilson confidence interval

The default critical value is `z=1.96` for a nominal 95% interval. The interval uses trapped and scored counts.
Per-disease results use the same calculation. No scored cases produce `NaN` rates and interval bounds.

### Length confound rate

This measures selection of the maximum candidate word count. Equal-length reports also count as selecting the maximum.
It does not isolate a preference for verbosity by itself. A high trap rate does not establish that jargon caused the errors.

## Prompts

Templates are defined in `core/prompts.py`. The writer is asked to describe the incorrect condition confidently.
Its default requested length is 90 to 130 words. The judge receives both reports and the image. It requests `Better report: A` or `Better report: B`
followed by a sentence of justification. Record prompt changes when comparing experiments.

## Image processing

Inputs are converted to RGB and encoded as JPEG. The default quality is 88. Large images are downsampled with Lanczos to at most 768 pixels
on their longest dimension. Aspect ratio is preserved, and smaller images are not upscaled. The judge receives a base64 JPEG image data URL.

## API integration

The client uses OpenRouter's chat-completions endpoint. Judge temperature defaults to 0.2; writer temperature to 0.7.
Each request defaults to a 60-second timeout and three retries. HTTP 429, selected 5xx errors, and network failures are retried.
Other client errors are reported immediately. Low temperature does not guarantee identical responses.

## Core function examples

Core helpers can be used without starting the interface.

```python
from core import parse_judge_verdict, evaluate_trap_result, wilson_ci

parsed = parse_judge_verdict("Better report: B\nMatches the lesion.")
result = evaluate_trap_result(parsed["pick"], correct_label="A")
print(result["status"])  # TRAPPED
print(wilson_ci(k=3, n=10))
```

## Sample data

Run `python sample_data/create_samples.py` to regenerate six cases. It overwrites generated CSV and PNG files under `sample_data/`.
The drawings include diagnostic text and are not clinical photographs. Use them to exercise the workflow, not to measure clinical performance.

## Testing

Run the existing suite from the repository root:

```powershell
python -m pytest
```

Tests cover core helpers, mocked API behavior, and UI-related logic. They also cover aliases, result coercion, and stable batch ordering.
Mocked tests do not verify live model availability.

## Troubleshooting

| Problem | Check |
| --- | --- |
| Missing package | Install requirements with the active interpreter. |
| Disabled evaluation | Supply a key, image, and both reports. |
| Authentication error | Replace the placeholder API key. |
| HTTP 402 | Review the insufficient-credit message. |
| HTTP 404 | Check the model slug or choose a custom model. |
| Rate limit | Reduce batch size or increase the case delay. |
| Missing image | Check the folder, filename, or ZIP entry. |
| Rejected CSV | Check required headers and nonempty fields. |
| Checkpoint failure | Use an existing writable parent directory. |
| UNCLEAR response | Inspect raw output and answer formatting. |

## Experiment practices

Review ground truth and incorrect targets before evaluation. Freeze jargon reports when comparing multiple judge models.
Keep stable identifiers and inspect the actual A/B distribution. Save input data, outputs, prompts, settings, and dependency versions.
Use separate checkpoint files for different experiments.

## Development

Keep reusable API and scoring logic in `core/`. Keep interface presentation and interactions in `ui/`.
Update relevant existing tests when changing supported behavior.

## Limitations

The application does not independently verify diagnoses. It does not enforce equal report lengths or exact A/B balance.
It has no built-in repeated-trial scheduler or complete run manifest. Dataset quality and external model behavior limit reproducibility.
The application is not a medical diagnosis system.

## Data handling

Judging sends images and both reports to remote model services. Writer calls send conditions and any additional text context.
Checkpoints store report text, image references, and responses. Use data appropriate for these transfers and review shared exports.
Keep API keys out of committed files and experiment artifacts.
