FROM node:22-alpine AS build
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY assets /app/assets
COPY web ./
ARG VITE_MAP_STYLE_URL=https://tiles.openfreemap.org/styles/positron
ENV VITE_MAP_STYLE_URL=$VITE_MAP_STYLE_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/web/dist /usr/share/nginx/html
