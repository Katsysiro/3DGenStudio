# Mesh2Motion (https://github.com/Mesh2Motion/mesh2motion-app) -- the browser
# tool for rigging and animating a mesh. The Docker counterpart of
# install-mesh2motion.bat, which clones the repo and runs `npm run dev`; here
# the app is built once and served as static files by nginx, which is what a
# server should run instead of the Vite dev server.
FROM node:24-alpine AS build
RUN apk add --no-cache git
ARG MESH2MOTION_REF=653a9698f5f315523072f1c3ff3496100401317b
WORKDIR /app
RUN git init -q . && \
    git fetch -q --depth 1 https://github.com/Mesh2Motion/mesh2motion-app "$MESH2MOTION_REF" && \
    git checkout -q FETCH_HEAD
RUN npm ci && npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
