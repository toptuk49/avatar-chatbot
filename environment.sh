#!/bin/bash
set -e

bw sync

BW_SESSION=$(bw unlock --raw)
export BW_SESSION

# --- PostgreSQL ---
DB_NAME=spintech_history
DB_LOGIN=$(bw get item "Spintech History Postgres Credentials" | jq -r ".login.username")
DB_PASSWORD=$(bw get item "Spintech History Postgres Credentials" | jq -r ".login.password")

# --- Эмбеддинг-модель ---
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DIM=768

# --- Языковая модель ---
# OpenRouter только для разработки!
OPENROUTER_API_KEY=$(bw get item "OpenRouter API Key" | jq -r ".notes")
OPENROUTER_MODEL=nvidia/nemotron-3-nano-30b-a3b:free

cat >.env <<EOF
# --- PostgreSQL ---
DB_NAME=$DB_NAME
DB_LOGIN=$DB_LOGIN
DB_PASSWORD=$DB_PASSWORD
DB_PORT=$DB_PORT

# --- Эмбеддинг-модель ---
EMBEDDING_MODEL=$EMBEDDING_MODEL
EMBEDDING_DIM=$EMBEDDING_DIM

# --- Языковая модель ---
# OpenRouter только для разработки!
OPENROUTER_API_KEY=$OPENROUTER_API_KEY
OPENROUTER_MODEL=$OPENROUTER_MODEL
EOF

echo "Created .env file with necessary secrets!"
