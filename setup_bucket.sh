#!/bin/bash

# Create MLflow bucket in MinIO
echo "Setting up MLflow bucket in MinIO..."

# Wait for MinIO to be ready
echo "Waiting for MinIO to start..."
sleep 10

# Configure mc (MinIO client) with our MinIO server
docker run --rm --network p2_default \
    -v $(pwd):/workspace \
    --entrypoint="" \
    minio/mc:latest \
    sh -c "
        echo 'Configuring MinIO client...'
        mc config host add minio http://mlflow-minio:9000 minioadmin minioadmin123 --api S3v4
        
        echo 'Creating mlflow-artifacts bucket...'
        mc mb minio/mlflow-artifacts --ignore-existing
        
        echo 'Setting bucket policy to public download...'
        mc anonymous set download minio/mlflow-artifacts
        
        echo 'Listing buckets...'
        mc ls minio/
        
        echo 'MinIO setup complete!'
    "

echo "MLflow bucket setup completed!"
echo "You can now access:"
echo "- MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)"
echo "- MLflow UI: http://localhost:5000"