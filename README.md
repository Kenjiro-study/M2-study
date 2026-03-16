**実験環境の立ち上げ方**:
- コマンドパレットで「Dev Containers: Rebuild Container」を選択して立ち上がるのを待つ
- tmuxでollama用のターミナルを作り, ollama serveでLLMを立ち上げて放置(これでLLMが使える) ← これはollama使ってた時の
- pip install vllm
- export CUDA_VISIBLE_DEVICES=1　で使用するGPUの番号を指定
- tmuxでvllm用のターミナルを作り, vllm serve openai/gpt-oss-20b --served-model-name gpt-oss-20b --gpu-memory-utilization 0.8でLLMを立ち上げて放置
- python3 stopword_set.pyを実行してstopwordsとpunkt_tabをダウンロード
- 実験実行！


**被験者実験の実行方法**:
- curl -L https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz -o ngrok.tgzでダウンロード
- tar -xvzf ngrok.tgzで解凍
- ./ngrok config add-authtoken “自分のトークン” で認証
- tmuxでapp_gradio.pyを実行
- 別のtmuxでapp_gradioので指定したポート番号を使用して, ./ngrok http “ポート番号”で外に公開！！！
- https://marnie-nonmuscular-annamarie.ngrok-free.devからアクセスできる！！

**Agent vs. Agentの実験の実行方法**:
- "python3 -m archive.comparison.run_experiments --output ./experiments --config baseline --name experiment1 --debug"を実行
