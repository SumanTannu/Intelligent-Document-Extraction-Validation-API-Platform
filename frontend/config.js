"use strict";

window.INTELLIDOC_CONFIG = Object.freeze({
    apiBaseUrl: ["localhost", "127.0.0.1"].includes(window.location.hostname)
        ? "http://localhost:8000"
        : "https://tannu-intellidoc-backend.onrender.com",
});
