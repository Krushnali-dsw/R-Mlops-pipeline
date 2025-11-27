# PowerShell script to create MLflow bucket in MinIO
Write-Host "Setting up MLflow bucket in MinIO..." -ForegroundColor Green

# Wait for MinIO to be ready
Write-Host "Waiting for MinIO to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Create bucket using mc in Docker
Write-Host "Creating MLflow bucket..." -ForegroundColor Yellow

docker run --rm --network p2_default `
    -v ${PWD}:/workspace `
    --entrypoint="" `
    minio/mc:latest `
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

Write-Host "MLflow bucket setup completed!" -ForegroundColor Green
Write-Host "You can now access:" -ForegroundColor Cyan
Write-Host "- MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)" -ForegroundColor White
Write-Host "- MLflow UI: http://localhost:5000" -ForegroundColor White