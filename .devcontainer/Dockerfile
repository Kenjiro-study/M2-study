# NVIDIAの公式CUDAイメージをベースにする
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# --- ここからPythonバージョンの変更 ---

# 必要なシステムライブラリとPython 3.11をインストール
RUN apt-get update && \
    # PPAを追加するために必要なパッケージをインストール
    apt-get install -y software-properties-common curl && \
    # deadsnakes PPAを追加 (新しいPythonバージョンを提供)
    add-apt-repository ppa:deadsnakes/ppa && \
    # パッケージリストを再度更新
    apt-get update && \
    # Python 3.11と関連ツール、gitをインストール
    apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        git \
        iproute2 \
        tmux \
    && rm -rf /var/lib/apt/lists/*

# システムのデフォルトのpython3を3.11に設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# pipを最新版にアップグレード
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.11

# --- ここからOllamaのインストールを追加 ---
RUN curl -fsSL https://ollama.com/install.sh | sh
# --- ここまで追加 ---

# requirements.txtを先に一時的な場所にコピーしてインストール
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 作業ディレクトリを作成
WORKDIR /workspace