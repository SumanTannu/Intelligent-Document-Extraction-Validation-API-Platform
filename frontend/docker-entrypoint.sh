#!/bin/sh
set -eu

: "${API_BASE_URL:?API_BASE_URL must be set to the backend Cloud Run URL}"

case "$API_BASE_URL" in
    http://*|https://*) ;;
    *) echo "API_BASE_URL must start with http:// or https://" >&2; exit 1 ;;
esac

api_base_url=${API_BASE_URL%/}
cat > /usr/share/nginx/html/config.js <<EOF
"use strict";

window.INTELLIDOC_CONFIG = Object.freeze({
    apiBaseUrl: "${api_base_url}",
});
EOF
