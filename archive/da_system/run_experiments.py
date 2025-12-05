# run_experiments.py
"""
AgreeMate 交渉実験を実行するためのメインエントリーポイント
コマンドラインインタフェース, ログ設定, 及び実験実行を処理する
"""
import sys, logging, asyncio, argparse, warnings
from pathlib import Path
from datetime import datetime

from .experiment_runner import ExperimentRunner
from .config import EXPERIMENT_CONFIGS, validate_config

warnings.filterwarnings("ignore", message="Failed to use structured output format, falling back to JSON mode.")

# logging を構成する
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """実験の結果を出力するディレクトリを作成して設定"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"

    # ディレクトリ構造を作成する
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    # ファイルへのログ記録の設定
    log_path = exp_dir / "logs" / "experiment.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    return exp_dir


async def run_experiment(args):
    """与えられた configuration で実験を実行する"""
    # configuration を検証する
    if args.config not in EXPERIMENT_CONFIGS:
        available = ", ".join(EXPERIMENT_CONFIGS.keys())
        logger.error(f"Unknown configuration '{args.config}'. Available: {available}")
        sys.exit(1)

    config = EXPERIMENT_CONFIGS[args.config]
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Invalid configuration: {str(e)}")
        sys.exit(1)

    # 実験(結果用?)ディレクトリのセットアップ
    exp_name = args.name or args.config
    exp_dir = setup_experiment_dir(args.output, exp_name)
    logger.info(f"Running experiment '{exp_name}' in {exp_dir}")

    try:
        # 実験を初期化して実行する
        runner = ExperimentRunner(
            config_name=args.config,
            output_dir=str(exp_dir),
            experiment_name=exp_name
        )

        results = await runner.run()

        # ログの completion summary
        logger.info("Experiment completed successfully:")
        logger.info(f"Total scenarios: {results['results'].scenarios_total}")
        logger.info(f"Completed: {results['results'].scenarios_completed}")
        logger.info(f"Failed: {results['results'].scenarios_failed}")
        logger.info(f"Results saved to: {exp_dir / 'results'}")

        return results

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="Run AgreeMate negotiation experiments"
    )

    # 必須のコマンドライン引数
    parser.add_argument(
        "--output",
        required=True,
        help="Base directory for experiment outputs"
    )

    # オプションのコマンドライン引数
    parser.add_argument(
        "--config",
        default="baseline",
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiment configuration to use"
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Custom name for this experiment run"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # 要求に応じてデバッグのログを設定する
    if args.debug:
        #logger.setLevel(logging.DEBUG)
        logging.getLogger("LiteLLM").setLevel(logging.WARNING) 
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)

    # 実験を実行する
    try:
        asyncio.run(run_experiment(args))
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()