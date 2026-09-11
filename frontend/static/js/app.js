"use strict";

const DOCUMENT_TYPE_LABELS = Object.freeze({
    invoice: "Invoice",
    balance_sheet: "Balance Sheet",
    profit_and_loss: "Profit and Loss",
    cash_flow_statement: "Cash Flow Statement",
});

const LOWERCASE_LABEL_WORDS = new Set([
    "and",
    "for",
    "in",
    "of",
    "or",
    "the",
    "to",
]);

const FRIENDLY_ERRORS = Object.freeze({
    400: "The file could not be validated. Check its format, contents, and PDF page count.",
    404: "The requested processed document could not be found.",
    409: "A document with this file name has already been processed.",
    422: "Text could not be extracted from this document.",
    429: "The extraction service is busy. Please wait and try again.",
    502: "Structured extraction could not be completed. Please try again.",
    503: "The service or document database is temporarily unavailable.",
});

document.addEventListener("DOMContentLoaded", () => {
    if (document.body.dataset.page === "dashboard") {
        initializeDashboard();
    } else if (document.body.dataset.page === "result") {
        initializeResultPage();
    }
});

function initializeDashboard() {
    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("document-file");
    const selectedFile = document.getElementById("selected-file");
    const refreshButton = document.getElementById("refresh-documents");

    fileInput.addEventListener("change", () => {
        selectedFile.textContent = fileInput.files.length
            ? fileInput.files[0].name
            : "Choose a supported financial document.";
    });

    form.addEventListener("submit", processDocument);
    refreshButton.addEventListener("click", loadDocuments);
    loadDocuments();
}

async function processDocument(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const fileInput = document.getElementById("document-file");
    const documentType = document.getElementById("document-type");
    const errorBox = document.getElementById("api-error");

    hideMessage(errorBox);
    if (!fileInput.files.length || !documentType.value) {
        showMessage(errorBox, "Choose a file and document type before processing.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("document_type", documentType.value);
    setProcessingState(true);

    try {
        const response = await fetch("/api/v1/documents/process", {
            method: "POST",
            body: formData,
        });
        const payload = await parseJsonResponse(response);
        if (!response.ok) {
            throw new ApiRequestError(response.status, payload);
        }
        if (!payload || typeof payload.file_name !== "string") {
            throw new Error("The server returned an incomplete processing response.");
        }
        window.location.assign(`/documents/${encodeURIComponent(payload.file_name)}`);
    } catch (error) {
        showMessage(errorBox, userMessageForError(error));
    } finally {
        setProcessingState(false);
    }
}

async function loadDocuments() {
    const loading = document.getElementById("documents-loading");
    const list = document.getElementById("documents-list");
    const errorBox = document.getElementById("documents-error");
    loading.hidden = false;
    loading.textContent = "Loading processed documents…";
    list.replaceChildren();
    hideMessage(errorBox);

    try {
        const response = await fetch("/api/v1/documents", {
            headers: { Accept: "application/json" },
        });
        const payload = await parseJsonResponse(response);
        if (!response.ok) {
            throw new ApiRequestError(response.status, payload);
        }

        const documents = Array.isArray(payload?.documents) ? payload.documents : [];
        if (!documents.length) {
            loading.textContent = "No documents have been processed yet.";
            return;
        }

        loading.hidden = true;
        documents.forEach((item) => list.append(createDocumentCard(item)));
    } catch (error) {
        loading.hidden = true;
        showMessage(errorBox, userMessageForError(error));
    }
}

function createDocumentCard(item) {
    const card = document.createElement("article");
    card.className = "document-card";

    const status = document.createElement("span");
    status.className = "status-badge status-pass";
    status.textContent = "Processed";

    const title = document.createElement("h3");
    title.textContent = typeof item.file_name === "string" ? item.file_name : "Unnamed document";

    const metadata = document.createElement("p");
    const typeLabel = DOCUMENT_TYPE_LABELS[item.document_type] || "Unknown document type";
    const validation = item.financial_validation?.status;
    metadata.textContent = validation
        ? `${typeLabel} · Financial validation: ${validation}`
        : typeLabel;

    const link = document.createElement("a");
    link.className = "document-link";
    link.textContent = "View result";
    link.href = `/documents/${encodeURIComponent(item.file_name || "")}`;

    card.append(status, title, metadata, link);
    return card;
}

async function initializeResultPage() {
    const documentName = document.body.dataset.documentName || "";
    const loading = document.getElementById("result-loading");
    const errorBox = document.getElementById("result-error");

    try {
        const response = await fetch(
            `/api/v1/documents/${encodeURIComponent(documentName)}`,
            { headers: { Accept: "application/json" } },
        );
        const payload = await parseJsonResponse(response);
        if (!response.ok) {
            throw new ApiRequestError(response.status, payload);
        }
        renderResult(payload);
        loading.hidden = true;
        document.getElementById("result-content").hidden = false;
    } catch (error) {
        loading.hidden = true;
        showMessage(errorBox, userMessageForError(error));
        setStatusBadge(document.getElementById("overall-status"), "FAIL");
        setStatusBadge(
            document.getElementById("header-financial-status"),
            "NOT AVAILABLE",
        );
    }
}

function renderResult(result) {
    const payload = isRecord(result) ? result : {};
    const validation = isRecord(payload.file_validation) ? payload.file_validation : {};
    const extraction = isRecord(payload.text_extraction) ? payload.text_extraction : {};
    const financial = isRecord(payload.financial_validation) ? payload.financial_validation : null;
    const documentType = DOCUMENT_TYPE_LABELS[payload.document_type]
        || displayValue(payload.document_type);
    const processingStatus = extraction.status === "PASS" ? "COMPLETED" : "FAIL";
    const financialStatus = financial?.status || "NOT AVAILABLE";

    setText("result-file-name", displayValue(payload.file_name));
    setText("summary-file-name", displayValue(payload.file_name));
    setText("summary-document-type", documentType);
    setText("summary-status", processingStatus);
    setText("summary-financial-status", formatStatusLabel(financialStatus));
    setStatusBadge(document.getElementById("overall-status"), processingStatus);
    setStatusBadge(document.getElementById("header-financial-status"), financialStatus);

    setStatusBadge(document.getElementById("validation-status"), validation.status);
    renderDefinitionList("validation-details", [
        ["File type", validation.file_type],
        ["Supported", formatBoolean(validation.is_supported)],
        ["Readable", formatBoolean(validation.is_readable)],
        ["Page count", validation.page_count],
    ]);
    const validationError = document.getElementById("file-validation-error");
    if (typeof validation.error === "string" && validation.error.trim()) {
        validationError.textContent = validation.error;
        validationError.hidden = false;
    } else {
        validationError.textContent = "";
        validationError.hidden = true;
    }

    setStatusBadge(document.getElementById("text-status"), extraction.status);
    renderDefinitionList("text-details", [
        ["Method", formatExtractionMethod(extraction.extraction_method)],
        ["Page count", extraction.page_count],
        ["Characters extracted", typeof extraction.extracted_text === "string" ? extraction.extracted_text.length : null],
    ]);
    renderTextExtractionDetails(extraction);
    renderStructuredExtraction(payload.structured_extraction);
    renderFinancialValidation(financial);

    document.getElementById("raw-response-json").textContent = safeJson(payload);
}

function renderStructuredExtraction(structuredExtraction) {
    const container = document.getElementById("extracted-fields");
    const emptyState = document.getElementById("structured-empty");
    container.replaceChildren();

    if (!isRecord(structuredExtraction)) {
        showInlineEmpty(emptyState, "Structured extraction is not available.");
        return;
    }

    const fields = Array.isArray(structuredExtraction.extracted_fields)
        ? structuredExtraction.extracted_fields.filter(isRecord)
        : [];
    if (!fields.length) {
        showInlineEmpty(emptyState, "No structured fields were returned.");
        return;
    }

    emptyState.hidden = true;
    fields.forEach((field) => container.append(createExtractedFieldCard(field)));
}

function createExtractedFieldCard(field) {
    const card = document.createElement("article");
    card.className = "extracted-field-card";
    const isNotApplicable = field.value === "NOT_APPLICABLE";

    const heading = document.createElement("div");
    heading.className = "extracted-field-heading";
    const name = document.createElement("h3");
    name.textContent = formatFieldName(field.field);
    heading.append(name);

    if (isNotApplicable) {
        const badge = document.createElement("span");
        setStatusBadge(badge, "NOT_APPLICABLE");
        heading.append(badge);
    }

    const value = document.createElement("p");
    value.className = isNotApplicable
        ? "extracted-value value-not-applicable"
        : "extracted-value";
    value.textContent = isNotApplicable ? "Not applicable" : displayValue(field.value);

    const confidence = document.createElement("p");
    confidence.className = "confidence-label";
    const confidenceName = document.createElement("span");
    confidenceName.textContent = "Extraction confidence";
    const confidenceValue = document.createElement("strong");
    confidenceValue.textContent = formatConfidence(field.confidence);
    confidence.append(confidenceName, confidenceValue);

    card.append(heading, value, confidence, renderEvidence(field.evidence, isNotApplicable));
    return card;
}

function formatFieldName(fieldName) {
    if (typeof fieldName !== "string" || !fieldName.trim()) {
        return "Unnamed field";
    }
    const tokens = fieldName.trim().split(/_+/).filter(Boolean);
    return tokens.map((token, index) => {
        if (/^\d+$/.test(token)) {
            return token;
        }
        const normalized = token.toLowerCase();
        if (index > 0 && LOWERCASE_LABEL_WORDS.has(normalized)) {
            return normalized;
        }
        return normalized.charAt(0).toUpperCase() + normalized.slice(1);
    }).join(" ");
}

function formatConfidence(confidence) {
    if (
        typeof confidence !== "number"
        || !Number.isFinite(confidence)
        || confidence < 0
        || confidence > 1
    ) {
        return "Not available";
    }
    return new Intl.NumberFormat("en", {
        style: "percent",
        maximumFractionDigits: 1,
    }).format(confidence);
}

function renderEvidence(evidenceItems, isNotApplicable = false) {
    const wrapper = document.createElement("div");
    wrapper.className = "evidence-wrapper";

    if (isNotApplicable) {
        const message = document.createElement("p");
        message.className = "empty-evidence";
        message.textContent = "Not applicable.";
        wrapper.append(message);
        return wrapper;
    }

    const evidence = Array.isArray(evidenceItems)
        ? evidenceItems.filter(isRecord)
        : [];
    if (!evidence.length) {
        const message = document.createElement("p");
        message.className = "empty-evidence";
        message.textContent = "Evidence is not available for this field.";
        wrapper.append(message);
        return wrapper;
    }

    const disclosure = document.createElement("details");
    disclosure.className = "evidence-disclosure";
    const summary = document.createElement("summary");
    summary.textContent = evidence.length === 1 ? "Evidence" : `Evidence (${evidence.length})`;
    disclosure.append(summary);

    evidence.forEach((item) => {
        const evidenceBlock = document.createElement("div");
        evidenceBlock.className = "evidence-item";
        const quotation = document.createElement("blockquote");
        quotation.textContent = typeof item.source_text === "string" && item.source_text
            ? `“${item.source_text}”`
            : "Source text is not available.";
        evidenceBlock.append(quotation);

        const metadata = document.createElement("div");
        metadata.className = "evidence-metadata";
        if (Number.isInteger(item.page_number) && item.page_number >= 1) {
            const page = document.createElement("span");
            page.textContent = `Page ${item.page_number}`;
            metadata.append(page);
        }
        const coordinates = formatBoundingBox(item.bounding_box);
        if (coordinates) {
            const coordinateLabel = document.createElement("span");
            coordinateLabel.textContent = `Coordinates: ${coordinates}`;
            metadata.append(coordinateLabel);
        }
        if (metadata.childElementCount) {
            evidenceBlock.append(metadata);
        }
        disclosure.append(evidenceBlock);
    });

    wrapper.append(disclosure);
    return wrapper;
}

function formatBoundingBox(boundingBox) {
    if (!isRecord(boundingBox)) {
        return "";
    }
    const keys = ["x", "y", "width", "height"];
    if (keys.some((key) => boundingBox[key] === null || boundingBox[key] === undefined)) {
        return "";
    }
    return keys.map((key) => `${key} ${displayValue(boundingBox[key])}`).join(", ");
}

function renderFinancialValidation(financialValidation) {
    const statusBadge = document.getElementById("financial-status");
    const periodsContainer = document.getElementById("validation-periods");
    const emptyState = document.getElementById("financial-empty");
    periodsContainer.replaceChildren();

    if (!isRecord(financialValidation)) {
        setStatusBadge(statusBadge, "NOT AVAILABLE");
        setText("financial-overall-status", "Not available");
        setText("financial-tolerance", "Not available");
        showInlineEmpty(emptyState, "Financial validation is not available.");
        return;
    }

    setStatusBadge(statusBadge, financialValidation.status);
    setText("financial-overall-status", formatStatusLabel(financialValidation.status));
    setText("financial-tolerance", displayValue(financialValidation.tolerance));

    const checks = Array.isArray(financialValidation.checks)
        ? financialValidation.checks.filter(isRecord)
        : [];
    if (!checks.length) {
        showInlineEmpty(emptyState, "No financial validation checks were applicable.");
        return;
    }

    emptyState.hidden = true;
    for (const [period, periodChecks] of groupChecksByPeriod(checks)) {
        const group = document.createElement("section");
        group.className = "validation-period";
        const heading = document.createElement("h3");
        heading.textContent = period === "General" ? period : formatFieldName(period);
        group.append(heading);

        const grid = document.createElement("div");
        grid.className = "validation-check-grid";
        periodChecks.forEach((check) => grid.append(createValidationCheck(check)));
        group.append(grid);
        periodsContainer.append(group);
    }
}

function groupChecksByPeriod(checks) {
    const groups = new Map();
    checks.forEach((check) => {
        if (!isRecord(check)) return;
        const period = typeof check.period === "string" && check.period.trim()
            ? check.period
            : "General";
        if (!groups.has(period)) {
            groups.set(period, []);
        }
        groups.get(period).push(check);
    });
    return groups;
}

function createValidationCheck(check) {
    const card = document.createElement("article");
    card.className = "validation-check";

    const heading = document.createElement("div");
    heading.className = "validation-check-heading";
    const name = document.createElement("h4");
    name.textContent = formatFieldName(check.rule_name);
    const badge = document.createElement("span");
    setStatusBadge(badge, check.status);
    heading.append(name, badge);
    card.append(heading);

    if (typeof check.formula === "string" && check.formula) {
        const formula = document.createElement("p");
        formula.className = "formula-display";
        const label = document.createElement("span");
        label.textContent = "Formula";
        const expression = document.createElement("code");
        expression.textContent = check.formula;
        formula.append(label, expression);
        card.append(formula);
    }

    const inputValues = renderInputValues(check.input_values);
    if (inputValues) {
        card.append(inputValues);
    }

    const results = [
        ["Calculated value", check.calculated_value],
        ["Reported value", check.reported_value],
        ["Variance", check.variance],
    ].filter(([, value]) => value !== null && value !== undefined && value !== "");
    if (results.length) {
        const resultList = document.createElement("dl");
        resultList.className = "check-results";
        results.forEach(([label, value]) => {
            const row = document.createElement("div");
            const term = document.createElement("dt");
            const description = document.createElement("dd");
            term.textContent = label;
            description.textContent = displayValue(value);
            row.append(term, description);
            resultList.append(row);
        });
        card.append(resultList);
    }

    const unavailableInputs = Array.isArray(check.unavailable_inputs)
        ? check.unavailable_inputs.filter((item) => typeof item === "string" && item)
        : [];
    if (unavailableInputs.length) {
        const section = document.createElement("div");
        section.className = "unavailable-inputs";
        const label = document.createElement("h5");
        label.textContent = "Unavailable inputs";
        const list = document.createElement("ul");
        unavailableInputs.forEach((item) => {
            const listItem = document.createElement("li");
            listItem.textContent = item;
            list.append(listItem);
        });
        section.append(label, list);
        card.append(section);
    }

    return card;
}

function renderInputValues(inputValues) {
    if (!isRecord(inputValues) || !Object.keys(inputValues).length) {
        return null;
    }

    const section = document.createElement("div");
    section.className = "input-values";
    const heading = document.createElement("h5");
    heading.textContent = "Input values";
    const table = document.createElement("table");
    const body = document.createElement("tbody");
    Object.entries(inputValues).forEach(([key, value]) => {
        const row = document.createElement("tr");
        const label = document.createElement("th");
        const displayedValue = document.createElement("td");
        label.scope = "row";
        label.textContent = formatFieldName(key);
        displayedValue.textContent = displayValue(value);
        row.append(label, displayedValue);
        body.append(row);
    });
    table.append(body);
    section.append(heading, table);
    return section;
}

function renderTextExtractionDetails(extraction) {
    setDisclosureText(
        "extracted-text-disclosure",
        "extracted-text-panel",
        extraction.extracted_text,
    );
    setDisclosureText(
        "raw-text-disclosure",
        "raw-text-panel",
        extraction.raw_text,
    );

    const ocrDisclosure = document.getElementById("ocr-details-disclosure");
    const ocrPanel = document.getElementById("ocr-details-panel");
    if (isRecord(extraction.ocr_details)) {
        ocrPanel.textContent = safeJson(extraction.ocr_details);
        ocrDisclosure.hidden = false;
    } else {
        ocrPanel.textContent = "";
        ocrDisclosure.hidden = true;
        ocrDisclosure.open = false;
    }
}

function setDisclosureText(disclosureId, panelId, value) {
    const disclosure = document.getElementById(disclosureId);
    const panel = document.getElementById(panelId);
    if (typeof value === "string" && value) {
        panel.textContent = value;
        disclosure.hidden = false;
    } else {
        panel.textContent = "";
        disclosure.hidden = true;
        disclosure.open = false;
    }
}

function showInlineEmpty(element, message) {
    element.textContent = message;
    element.hidden = false;
}

function displayValue(value, fallback = "—") {
    if (value === null || value === undefined || value === "") {
        return fallback;
    }
    if (typeof value === "boolean") {
        return value ? "Yes" : "No";
    }
    if (typeof value === "string" || typeof value === "number") {
        return String(value);
    }
    return "Unsupported value";
}

function formatExtractionMethod(method) {
    if (method === "native_text") return "Native PDF text";
    if (method === "ocr") return "OCR";
    return "—";
}

function formatStatusLabel(status) {
    return typeof status === "string" && status
        ? status.replaceAll("_", " ")
        : "Not available";
}

function safeJson(value) {
    try {
        return JSON.stringify(value, null, 2) || "No response data is available.";
    } catch {
        return "The response could not be displayed as JSON.";
    }
}

function isRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
}

function renderDefinitionList(elementId, entries) {
    const list = document.getElementById(elementId);
    list.replaceChildren();
    entries.forEach(([label, value]) => {
        const row = document.createElement("div");
        const term = document.createElement("dt");
        const description = document.createElement("dd");
        term.textContent = label;
        description.textContent = displayValue(value);
        row.append(term, description);
        list.append(row);
    });
}

function setStatusBadge(element, status) {
    const normalized = typeof status === "string" ? status.toUpperCase() : "NOT AVAILABLE";
    element.textContent = normalized.replaceAll("_", " ");
    element.className = "status-badge status-neutral";
    if (["PASS", "COMPLETED", "PROCESSED"].includes(normalized)) {
        element.className = "status-badge status-pass";
    } else if (normalized === "FAIL") {
        element.className = "status-badge status-fail";
    } else if (normalized === "NOT_APPLICABLE") {
        element.className = "status-badge status-na";
    }
}

function setProcessingState(isProcessing) {
    const button = document.getElementById("process-button");
    const state = document.getElementById("processing-state");
    button.disabled = isProcessing;
    button.classList.toggle("is-loading", isProcessing);
    button.querySelector(".button-label").textContent = isProcessing
        ? "Processing…"
        : "Process document";
    state.hidden = !isProcessing;
}

async function parseJsonResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
        return null;
    }
    try {
        return await response.json();
    } catch {
        return null;
    }
}

function userMessageForError(error) {
    if (error instanceof ApiRequestError) {
        const validationMessage = error.payload?.detail?.file_validation?.error;
        if (typeof validationMessage === "string" && validationMessage) {
            return validationMessage;
        }
        return FRIENDLY_ERRORS[error.status] || "The request could not be completed.";
    }
    if (error instanceof TypeError) {
        return "The server could not be reached. Check that IntelliDoc is running.";
    }
    return "The request could not be completed. Please try again.";
}

function showMessage(element, message) {
    element.textContent = message;
    element.hidden = false;
}

function hideMessage(element) {
    element.textContent = "";
    element.hidden = true;
}

function setText(elementId, value) {
    document.getElementById(elementId).textContent = String(value);
}

function formatBoolean(value) {
    if (value === true) return "Yes";
    if (value === false) return "No";
    return "—";
}

class ApiRequestError extends Error {
    constructor(status, payload) {
        super(`API request failed with status ${status}`);
        this.name = "ApiRequestError";
        this.status = status;
        this.payload = payload;
    }
}
