FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY frontend/docker-entrypoint.sh /docker-entrypoint.d/40-intellidoc-config.sh
COPY frontend/index.html frontend/document_result.html frontend/config.js /usr/share/nginx/html/
COPY frontend/static /usr/share/nginx/html/static

RUN chmod +x /docker-entrypoint.d/40-intellidoc-config.sh

EXPOSE 8080
