FROM node:22-slim AS builder

WORKDIR /workspace

COPY package.json yarn.lock tsconfig.base.json ./
COPY packages/social-pet-domain/package.json packages/social-pet-domain/package.json
COPY services/social-pet-api/package.json services/social-pet-api/package.json

RUN yarn install --frozen-lockfile

COPY packages/social-pet-domain packages/social-pet-domain
COPY services/social-pet-api services/social-pet-api

RUN yarn workspace @social-pet/domain build
RUN yarn workspace @social-pet/api build

FROM node:22-slim AS runner

ENV NODE_ENV=production
ENV PORT=8080

WORKDIR /app

COPY package.json yarn.lock ./
COPY packages/social-pet-domain/package.json packages/social-pet-domain/package.json
COPY services/social-pet-api/package.json services/social-pet-api/package.json

RUN yarn install --frozen-lockfile --production=true --ignore-scripts

COPY --from=builder /workspace/packages/social-pet-domain/dist /app/packages/social-pet-domain/dist
COPY --from=builder /workspace/services/social-pet-api/dist /app/services/social-pet-api/dist

EXPOSE 8080

CMD ["node", "services/social-pet-api/dist/app.js"]
