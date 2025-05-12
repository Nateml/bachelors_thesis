import subprocess
import sys

params = [
    {
        "num_cnn_layers": 3,
        "in_channels": [1, 32, 64],
        "out_channels": [32, 64, 128],
        "kernel_sizes": [7, 5, 3],
        "strides": [1, 1, 1],
        "paddings": [3, 2, 1]
    },
    {
        "num_cnn_layers": 3,
        "in_channels": [1, 32, 64],
        "out_channels": [32, 64, 128],
        "kernel_sizes": [11, 7, 3],
        "strides": [1, 1, 1],
        "paddings": [5, 3, 1]
    },
    {
        "num_cnn_layers": 4,
        "in_channels": [1, 32, 64, 128],
        "out_channels": [32, 64, 128, 256],
        "kernel_sizes": [11, 7, 5, 3],
        "strides": [1, 1, 1, 1],
        "paddings": [5, 3, 2, 1]
    },
    {
        "num_cnn_layers": 4,
        "in_channels": [1, 32, 64, 128],
        "out_channels": [32, 64, 128, 256],
        "kernel_sizes": [17, 11, 7, 5],
        "strides": [1, 1, 1, 1],
        "paddings": [8, 5, 3, 2]
    },
    {
        "num_cnn_layers": 5,
        "in_channels": [1, 32, 64, 128, 256],
        "out_channels": [32, 64, 128, 256, 512],
        "kernel_sizes": [11, 9, 7, 5, 3],
        "strides": [1, 1, 1, 1],
        "paddings": [5, 4, 3, 2, 1]
    },
    {
        "num_cnn_layers": 5,
        "in_channels": [1, 32, 64, 128, 256],
        "out_channels": [32, 64, 128, 256, 512],
        "kernel_sizes": [17, 11, 7, 5, 3],
        "strides": [1, 1, 1, 1],
        "paddings": [8, 5, 3, 2, 1]
    },
]

fixed_params = {
    "light_dropout": 0.1,
    "heavy_dropout": 0.2,
}

PYTHON = sys.executable

base_cmd = [
    PYTHON,
    "bachelors_thesis/run.py",
    "loss=cross_entropy",
    "model=siglabv2",
    "model.encoder=simplecnn",
    "run.batch_size=38",
    "optimizer.lr=7e-3",
    "run.patience=10",
    "run.epochs=50",
    "run.name=simplecnn_sweep"
]


# Run the experiment with the specified parameters
def run_experiment(params):
    for param in params:
        cmd = base_cmd + [
            f"model.encoder.{key}={value}" for key, value in param.items()
        ]

        print(f"Running command: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    all_params = params.copy()
    all_params.append(fixed_params)
    run_experiment(all_params)
