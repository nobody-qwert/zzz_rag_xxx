FROM node:18-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install --no-audit --fund=false
COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
# Nginx will serve the built static files and proxy /api to the backend service.
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
