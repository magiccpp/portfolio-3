import argparse
import os
import time
import requests
import joblib
import numpy as np
import threading
from typing import Dict
from sklearn.base import BaseEstimator
from collections import defaultdict

from kserve import (
    Model,
    ModelServer,
    InferRequest,
    InferResponse,
    model_server
)

from kserve.utils.utils import get_predict_response
import logging

# optional: basic python logging config
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))

class DynamicSklearnGRPCModel(Model):
    """
    A KServe custom model that speaks gRPC-V2 (InferenceProtocol.V2).

    Expects two inputs:
      - name="model_name", datatype="BYTES", shape=[1]
      - name="features",   datatype="FP32",  shape=[batch_size, n_features]
    """

    def __init__(
        self,
        name: str,
        model_storage_url: str,
        cache_dir: str = "/tmp/models_cache/"
    ):
        # return_response_headers=True to get the response_headers dict passed in
        super().__init__(name, return_response_headers=True)
        self.model_storage_url = model_storage_url.rstrip("/")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model_cache: Dict[str, BaseEstimator] = {}
        self.model_locks = defaultdict(threading.Lock)
        self.ready = False
        self._mark_ready()

    def _mark_ready(self):
        # Called once at init to signal readiness probe
        self.ready = True
        logging.info(f"Model `{self.name}` is ready for gRPC v2 requests.")

    def get_model_path(self, model_name: str) -> str:
        return os.path.join(self.cache_dir, f"{model_name}.joblib")

    def download_model(self, model_name: str) -> str:
        url = f"{self.model_storage_url}/{model_name}.joblib"
        logging.info(f"Downloading `{model_name}` from {url}")
        res = requests.get(url)
        if res.status_code != 200:
            raise FileNotFoundError(f"Could not download `{model_name}` ({res.status_code})")
        path = self.get_model_path(model_name)
        with open(path, "wb") as f:
            f.write(res.content)
        return path

    def load_model(self, model_name: str) -> BaseEstimator:
        if model_name in self.model_cache:
            logging.info(f"Model `{model_name}` fetched from cache.")
            return self.model_cache[model_name]
        lock = self.model_locks[model_name]
        with lock:
            path = self.get_model_path(model_name)
            if not os.path.isfile(path):
                path = self.download_model(model_name)
            model = joblib.load(path)
            self.model_cache[model_name] = model
            logging.info(f"Loaded `{model_name}` into cache.")
            return model

    async def predict(
        self,
        request: InferRequest,
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> InferResponse:
        logging.info(f"Received gRPC v2 request for model `{self.name}`")
        """
        request.inputs is a list of InferenceRequestInput protos.
        We expect:
          inputs[0].name == "model_name"   datatype="BYTES"
          inputs[1].name == "features"     datatype="FP32"
        """
        start = time.time()

        # 1) extract model_name

        mn_input = request.inputs[0]
        arr = mn_input.as_numpy()
        b = arr[0]                      # this is a bytes object
        model_name = b.decode("utf-8")
        if mn_input.datatype != "BYTES":
            raise ValueError("The first input must be BYTES model_name")

        # 2) extract feature tensor
        feat_input = request.inputs[1]
        if feat_input.datatype != "FP32":
            raise ValueError("The second input must be FP32 features")
        features = feat_input.as_numpy()  # shape [batch, n_features]
        if features.ndim == 1:
            # turn (n_features,) into (1, n_features)
            features = features.reshape(1, -1)
        # 3) load sklearn model
        model = self.load_model(model_name)

        # 4) predict
        preds = model.predict(features)  # could be shape [batch] or [batch, ...]
        # ensure numpy array
        preds = np.array(preds)

        # 5) latency header
        latency_ms = (time.time() - start) * 1000.0
        if response_headers is not None:
            response_headers["prediction-time-latency-ms"] = f"{latency_ms:.3f}"

        # 6) wrap up in a v2 InferResponse
        # get_predict_response will automatically name the output
        # after our model name, or you can pass a custom output name.
        return get_predict_response(request, preds, self.name)

if __name__ == "__main__":
    logging.info("Starting KServe gRPC V2 server...")
    parser = argparse.ArgumentParser(parents=[model_server.parser])

    parser.add_argument(
        "--model_storage_url",
        required=True,
        help="https://stockmodels.blob.core.windows.net/models/"
    )
    
    args, _ = parser.parse_known_args()
    # instantiate your V2 gRPC model
    model = DynamicSklearnGRPCModel(
        name=args.model_name,
        model_storage_url=args.model_storage_url,
    )

    # start both HTTP & gRPC servers
    ModelServer(workers=args.workers).start([model])