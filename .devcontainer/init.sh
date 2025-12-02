#!/bin/bash

# Ollamaサーバーをバックグラウンドで起動
echo "Starting Ollama server in background..."
ollama serve > /tmp/ollama.log 2>&1 &
PID=$!

# サーバーが起動するまで待機
echo "Waiting for Ollama server to be ready..."
while ! curl -s --head http://localhost:11434 | head -n 1 | grep "200 OK" > /dev/null; do
    sleep 1
done
echo "Ollama server is ready!"

# モデルをダウンロード（存在しない場合のみ）
echo "Pulling llama3.3:70b model..."
ollama pull llama3.3:70b

# 初期化が完了したことを通知し、サーバープロセスをフォアグラウンドで待機させる
echo "Initialization complete. Ollama server is running."
