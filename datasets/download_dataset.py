import kagglehub

# Download latest version
path = kagglehub.dataset_download("tolgadincer/labeled-chest-xray-images")  # It is 1.17G

print("Path to dataset files:", path)