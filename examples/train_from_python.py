from pathlib import Path

from cistempoforge import fit, load_config, load_model


config = load_config(Path(__file__).with_name("d0_d5_config.json"))
config.training.learning_rate = 1e-4

result = fit(config, output_dir="runs/d0_d5_seed42")
model, checkpoint = load_model(result.best_checkpoint, device="cuda")
print(result.best_epoch, checkpoint["metrics"])
