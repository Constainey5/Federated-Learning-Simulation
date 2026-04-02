
# Federated-Learning-Simulation

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-6200EE?style=for-the-badge&logo=n-a&logoColor=white)](https://en.wikipedia.org/wiki/Federated_learning)

A simulation framework for Federated Learning, demonstrating collaborative model training without sharing raw data.

## Overview

This repository provides a basic simulation of a Federated Learning (FL) setup. Federated Learning is a distributed machine learning approach that enables training on a decentralized dataset residing on various client devices, without the need to centralize the data. This enhances privacy and reduces communication overhead.

## Features

-   **Client-Server Architecture**: Simulates multiple clients and a central server.
-   **Local Training**: Clients train models on their local data.
-   **Model Aggregation**: Server aggregates client model updates.
-   **Iterative Learning**: Demonstrates multiple rounds of federated training.
-   **Privacy-Preserving**: No raw data leaves client devices.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

-   Python 3.8+
-   pip

### Installation

1.  Clone the repo
    ```bash
    git clone https://github.com/Constainey5/Federated-Learning-Simulation.git
    ```
2.  Install Python packages
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the `main.py` script to start the federated learning simulation.

```bash
python main.py
```

## Contributing

Contributions are welcome! Feel free to extend the simulation with more complex models, aggregation strategies, or client behaviors.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Dr. Constantine Reynolds - [https://www.linkedin.com/in/constantinereynoldsai](https://www.linkedin.com/in/constantinereynoldsai)

Project Link: [https://github.com/Constainey5/Federated-Learning-Simulation](https://github.com/Constainey5/Federated-Learning-Simulation)
