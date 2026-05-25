FROM node:18-alpine AS build

WORKDIR /app

# VITE_API_URL is resolved at build time (Vite inlines it into the bundle),
# so it must be supplied as a build arg — not a runtime env var.
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

# Install deps first so source-only changes don't bust this layer.
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage.
FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
