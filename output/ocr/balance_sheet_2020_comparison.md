# OCR comparison: Consolidated Balance Sheet 2020

## Source

`New Dataset/Balance Sheet/Consolidated Balance Sheet 2020.pdf`

The PDF contains a scanned page and no usable native text layer. The original-page transcription was therefore verified against a rendered image of the complete page.

## Results

| Check | Enhanced OCR | Original page | Result |
|---|---:|---:|---|
| Financial amount cells | 32 | 32 | 32/32 exact |
| Financial row labels | 14 | 14 | 14/14 exact |
| Main section headings | 2 | 2 | Exact |
| Schedule references | 14 | 14 | Values exact |
| Column dates | 2 | 2 | Exact |
| Signatory names | 6 | 6 | Exact |
| ICAI registration number | 1 | 1 | Exact |

## Content differences

There are no remaining financial-value differences in the checked page.

- The original page displays schedule `17 & 18`; OCR returns `17&18`. The value is the same, but spacing differs.
- The PDF uses columns and multi-column signature blocks. OCR flattens these into reading-order lines, so visual alignment differs even though the text content is retained.
- Font weight, italics, blue section styling, rules and borders are not represented in plain-text OCR output.

## Files

- `balance_sheet_2020_ocr.txt`: output returned by the enhanced OCR code.
- `balance_sheet_2020_original_transcription.txt`: visually verified original-page content in normalized plain-text layout.
