import subprocess
import sys


def run_step(description, command):
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)

    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        sys.exit(result.returncode)

    print(f"✅ COMPLETED: {description}")


def main():
    print("Northgate AI Stock Predictor")
    print("Reproducible Pipeline")
    print("=" * 70)

    python = sys.executable

    run_step(
        "1. Download raw market data",
        [python, "src/ingest.py"],
    )

    run_step(
        "2. Clean and align market data",
        [python, "src/clean.py"],
    )

    run_step(
        "3. Generate features",
        [python, "src/features.py"],
    )

    run_step(
        "4. Run classical ML evaluation",
        [python, "src/models_ml.py"],
    )

    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
