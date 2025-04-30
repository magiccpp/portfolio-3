import argparse
import base64
import io
import os
import time
import requests
from typing import Dict

from fastapi.middleware.cors import CORSMiddleware
from sklearn.base import BaseEstimator
import joblib

import kserve
from kserve import Model, ModelServer, logging
from kserve.model_server import app
from kserve.utils.utils import generate_uuid
import numpy as np
from util import save_pkl, load_pkl, get_pipline_svr, get_pipline_rf
from safeRegressors import SafeRandomForestRegressor, SafeSVR, TimeoutException

#python dynamic-model1.py --model_storage_url="https://stockmodels.blob.core.windows.net/models/" --http_port=8181 --grpc_port=8182
class DynamicSklearnModel(Model):
    def __init__(self, name: str, model_storage_url: str, cache_dir: str = "/tmp/models_cache/"):
        super().__init__(name, return_response_headers=True)
        self.name = name
        self.model_storage_url = model_storage_url.rstrip("/")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model_cache: Dict[str, BaseEstimator] = {}
        self.ready = False
        self.load()

    def load(self):
        # Initial load can be empty or load a default model if needed
        # Mark as ready
        self.ready = True
        print(f"DynamicSklearnModel '{self.name}' is ready.")

    def get_model_path(self, model_name: str) -> str:
        return os.path.join(self.cache_dir, f"{model_name}.joblib")

    def download_model(self, model_name: str) -> str:
        url = f"{self.model_storage_url}/{model_name}.joblib"
        print(f"Downloading model '{model_name}' from '{url}'")
        response = requests.get(url)
        if response.status_code == 200:
            model_path = self.get_model_path(model_name)
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Model '{model_name}' downloaded and saved to '{model_path}'")
            return model_path
        else:
            print(f"Failed to download model '{model_name}': HTTP {response.status_code}")
            raise FileNotFoundError(f"Model '{model_name}' not found in storage.")

    def load_model(self, model_name: str) -> BaseEstimator:
        if model_name in self.model_cache:
            print(f"Loading model '{model_name}' from cache.")
            return self.model_cache[model_name]

        model_path = self.get_model_path(model_name)
        if not os.path.exists(model_path):
            model_path = self.download_model(model_name)

        try:
            model = joblib.load(model_path)
            self.model_cache[model_name] = model
            print(f"Model '{model_name}' loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model '{model_name}': {str(e)}")
            raise e

    async def predict(
        self,
        payload: Dict,
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> Dict:
        start_time = time.time()

        try:
            # Extract the first instance
            instance = payload.get("instances", [{}])[0]
            model_name = instance.get("model_name")
            features = instance.get("features")

            if not model_name or features is None:
                raise ValueError("Request must contain 'model_name' and 'features'.")

            # Load the model (from cache or download)
            model = self.load_model(model_name)


            # Perform prediction
            prediction = model.predict(features).tolist()

            end_time = time.time()
            prediction_latency = round((end_time - start_time) * 1000, 3)  # in milliseconds

            # Add prediction latency to response headers
            if response_headers is not None:
                response_headers.update({"prediction-time-latency-ms": str(prediction_latency)})

            return {"predictions": prediction}

        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            raise e


if __name__ == "__main__":
    print("Starting KServe Dynamic Model Server...")
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    parser.add_argument(
        "--model_storage_url",
        type=str,
        required=True,
        help="Base URL for storing models (e.g., https://stockmodels.blob.core.windows.net/models/)",
    )
    args, _ = parser.parse_known_args()

    # Configure logging
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)

    # Initialize the custom model
    model = DynamicSklearnModel(
        name=args.model_name,
        model_storage_url=args.model_storage_url,
        cache_dir="/tmp/models_cache/"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust as needed for security
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Start the KServe model server with the custom model
    ModelServer().start([model])