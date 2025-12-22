# model_loader.py
import os, torch
from typing import Dict, Optional, Tuple
from pathlib import Path
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import MODEL_CONFIGS


class ModelLoader:
    """
    AgreeMate baseline system 用のモデル読み込み及びキャッシュユーティリティ
    事前学習済みモデルとトークナイザーの読み込み及びキャッシュを処理する
    """
    def __init__(self, cache_dir: Optional[str] = None):
        """オプションのキャッシュディレクトリを使用して ModelLoader を初期化する"""
        baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        agreemate_dir = os.path.dirname(baseline_dir)
        pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")
        self.cache_dir = cache_dir or pretrained_dir
        self.loaded_models = {}
        self.loaded_tokenizers = {}

        if not os.path.exists(self.cache_dir):
            raise ValueError(f"Cache directory does not exist: {self.cache_dir}")
        else:
            print(f"Using cache directory: {self.cache_dir}")

    @lru_cache(maxsize=3)
    def load_model_and_tokenizer(
        self,
        model_key: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16 # NVIDIA の例に従って bfloat16 を使用する
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        キャッシュを使用して, HuggingFaceからモデルとトークナイザーをロードする
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")

        config = MODEL_CONFIGS[model_key]

        #model_name = config["name"]
        model_name = config.name # 2025/7/15 変更

        try:
            # トークナイザーをロード
            if model_name not in self.loaded_tokenizers:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir # まずキャッシュをチェックする
                )
                self.loaded_tokenizers[model_name] = tokenizer

            # モデルをロード
            if model_name not in self.loaded_models:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir, # まずキャッシュをチェックする
                    torch_dtype=torch_dtype,
                    device_map="auto" # transformers に multi-GPU を処理させる
                )
                self.loaded_models[model_name] = model

            return self.loaded_models[model_name], self.loaded_tokenizers[model_name]

        except Exception as e:
            raise RuntimeError(f"Error loading model {model_key}: {str(e)}")

    def get_model_config(self, model_key: str) -> Dict:
        """指定されたモデルの configuration を取得する"""
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")
        return MODEL_CONFIGS[model_key].copy()

    async def generate_response(
        self,
        model_key: str,
        messages: list, # NVIDIA の例に示されているメッセージ形式を使用する
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        チャット形式を使用して指定されたモデルからの応答を生成する
        """
        model, tokenizer = self.load_model_and_tokenizer(model_key)
        config = self.get_model_config(model_key)

        # 指定されていない場合は config のデフォルトを使用する
        max_new_tokens = max_new_tokens or 4096 # NVIDIA の例からのデフォルト
        #temperature = temperature or config["temperature"]
        temperature = temperature or config.temperature # 2025/7/15 変更

        # モデルのチャットテンプレートを使用してチャットメッセージをフォーマットする
        tokenized_message = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )

        inputs = {
            'input_ids': tokenized_message['input_ids'].to(model.device),
            'attention_mask': tokenized_message['attention_mask'].to(model.device)
        }

        # 応答の生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # 新しいトークンのみを抽出する (response)
        generated_tokens = outputs[:, len(tokenized_message['input_ids'][0]):]
        response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return response.strip()


def model_loader():
    """モデルをロードし, トークナイズをテストし, 全てのモデルのキャッシュを実行する"""
    loader = ModelLoader()

    # config 内の全てのモデルを反復処理する
    for model_key, config in MODEL_CONFIGS.items():
        print(f"Loading and testing model: {model_key}")
        try:
            # model と tokenizer をロードする
            model, tokenizer = loader.load_model_and_tokenizer(model_key)

            # テストチャットメッセージを適用する
            test_message = [{"role": "user", "content": "How many r in strawberry?"}]
            tokenized = tokenizer.apply_chat_template(
                test_message,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            assert tokenized is not None, f"Tokenization failed for {model_key}"
            print(f"✓ {model_key} loaded, tokenized, and ready")
        except Exception as e:
            print(f"✗ Error testing {model_key}: {str(e)}")

    # キャッシュディレクトリが設定されていることを確認する
    pretrained_dir = loader.cache_dir
    cached_files = list(Path(pretrained_dir).glob("**/*"))
    if not cached_files:
        print("✗ Cache directory is empty!")
    else:
        print(f"✓ Cache directory populated with {len(cached_files)} files")

    print("All model tests complete.")

if __name__ == "__main__":
    loader = model_loader()