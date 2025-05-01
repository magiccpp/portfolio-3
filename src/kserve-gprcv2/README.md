https://kserve.github.io/website/0.14/modelserving/v1beta1/custom/custom_model/#implement-custom-model-using-kserve-api

run locally:
python dynamic-model1.py --model_storage_url="https://stockmodels.blob.core.windows.net/models/" --http_port=8181 --grpc_port=8182

build docker
pack build --builder=heroku/builder:24 magiccpp1/dynamic-model-grpc:v1
docker push magiccpp1/dynamic-model-grpc:v1
kubectl delete inferenceservice dynamic-model
kubectl apply -f dynamic.yaml


export INGRESS_HOST=$(kubectl get node -o jsonpath='{.items[0].status.addresses[0].address}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')
export SERVICE_HOSTNAME=$(kubectl get inferenceservice dynamic-model-grpcv2 -o jsonpath='{.status.url}' | cut -d "/" -f 3)

env|grep ING

copy above into kserv-dynamic-client.ipynb



