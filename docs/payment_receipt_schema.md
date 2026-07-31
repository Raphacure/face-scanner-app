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
