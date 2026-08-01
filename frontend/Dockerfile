# AcousticSpace Frontend - Week 4 (Deployment)
#
# Stage 1: build the static Vite bundle
# Stage 2: serve it with nginx — no Node runtime needed in the final image

FROM node:20-slim AS builder

WORKDIR /build

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build


FROM nginx:1.27-alpine

COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
