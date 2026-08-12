# Classify-receipt API

`POST /api/v1/documents/classify-receipt`

## Request

```json
{
  "urls": [
    { "url": "https://cdn.example.com/bill.jpg", "name": "invoice" },
    { "url": "https://cdn.example.com/rx.jpg", "name": "prescription" },
    { "url": "https://cdn.example.com/upi.png", "name": "payment_receipt" }
  ]
}
```

`name` is required: `invoice` | `prescription` | `report` | `payment_receipt`  
Aliases: `bill`, `rx`, `opd`, `lab`, `payment`, `upi`

Max 25 items. Order of `results[]` matches `urls[]`.

## Latency / timeouts

Typical per file: **15–35s** (OpenAI Vision). Invoice with Textract: **25–45s**.  
Batch time ≈ slowest file when `OPENAI_CLASSIFY_MAX_PARALLEL` ≥ file count (default **6**).

With CRM `name` set, Textract runs **only for `invoice`** (`TEXTRACT_HINT_INVOICE_ONLY=true`, default).

Many gateways timeout at **60s** — raise proxy timeout to **120–180s** for classify-receipt, or keep batches small (2–4 files).

Optional speed env vars:

- `OPENAI_CLASSIFY_MAX_PARALLEL=6` — parallel files (default 6)
- `OPENAI_IMAGE_DETAIL=low` — faster Vision (slightly less OCR on tiny text)
- `OPENAI_REFINE_PASSES=false` — keep off (default)
- `USE_TEXTRACT=false` — OpenAI-only (fastest, weaker printed GST/DL on bills)

Response includes `processing_time_ms` (batch + per result).

## Avoid gateway timeout (website / CRM)

Postman often works while the website gets **504 Gateway Timeout** because nginx/ALB/Cloudflare usually cut off at **60s**, but classify can take **45–90s**.

### Option A — Async (recommended for website)

**Step 1 — submit** (returns in <1 second):

```json
{
  "async": true,
  "urls": [
    { "url": "https://.../bill.jpg", "name": "invoice" },
    { "url": "https://.../rx.jpg", "name": "prescription" }
  ]
}
```

Response `202`:

```json
{
  "status": "accepted",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "poll_url": "/api/v1/documents/classify-receipt/jobs/550e8400-...",
  "poll_interval_ms": 2000
}
```

**Step 2 — poll** every 2s:

`GET /api/v1/documents/classify-receipt/jobs/{job_id}`

Until `status` is `success` (same body as sync) or `error`.

### Option B — Raise proxy timeout

On the gateway between website and Face-AI:

```nginx
proxy_read_timeout 180s;
proxy_connect_timeout 180s;
proxy_send_timeout 180s;
```

Also increase timeout on the CRM HTTP client calling Face-AI.

### Option C — Faster sync

- Keep batches small (2–4 files)
- `OPENAI_CLASSIFY_MAX_PARALLEL=6`
- CRM always sends `name` (skips Textract on non-invoice files)

## Downstream CRM / validation service

Face-AI **extracts** fields. Your **main service validates** and creates the claim.

### Flow (async — recommended)

```
Website → CRM API → Face-AI POST (async: true)  → 202 + job_id (< 1s)
CRM API → poll GET /jobs/{job_id}               → processing…
CRM API → poll until status = success           → full results
CRM API → validate each document                → accept / reject claim
```

Do **not** validate on `accepted` or `processing` — only when poll returns `status: "success"`.

### Matching request → response

`results[i]` matches `urls[i]`. Each result includes:

| Field | Use in validation |
| --- | --- |
| `name` | CRM label you sent |
| `document_category` | Should equal `name` |
| `parameters` | Extracted values |
| `missing_parameters` | Fields not readable |
| `completeness_percent` | Extraction quality hint |

### CRM validation rules

**invoice:** `patient_name`, `invoice_number`, `invoice_date`, `provider_name`, `total_amount`

**prescription:** `patient_name`, `consultation_date`, `clinic_hospital_name`, `doctor_name`, …

**payment_receipt:** `payment_amount`, `transaction_date`, (`transaction_id` OR `reference_number` OR `utr`), `payment_status` = success|completed

**report:** `patient_name`, `laboratory_name`, `test_names`, `test_results`, …
