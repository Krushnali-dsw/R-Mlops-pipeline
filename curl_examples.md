# 🚀 Model Deployment with Flask REST API - CURL Examples

## 📊 Your ML Model is Successfully Deployed!

**Server URL**: http://localhost:9090
**Model**: Random Forest Loan Approval Classifier
**Status**: ✅ Running and Ready

---

## 🔧 Available Endpoints

### 1. Health Check
```bash
curl -X GET "http://localhost:9090/health"
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:9090/health" -Method GET
```

**Expected Response:**
```json
{
  "model_loaded": true,
  "status": "healthy", 
  "version": "1.0.0"
}
```

---

### 2. Model Metadata
```bash
curl -X GET "http://localhost:9090/metadata"
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:9090/metadata" -Method GET
```

---

### 3. Make Predictions

#### 🎯 High Approval Probability Sample
```bash
curl -X POST "http://localhost:9090/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 35,
       "income": 75000,
       "education": 16,
       "experience": 10,
       "credit_score": 750
     }'
```

**PowerShell:**
```powershell
$body = '{"age":35,"income":75000,"education":16,"experience":10,"credit_score":750}'
Invoke-RestMethod -Uri "http://localhost:9090/predict" -Method POST -Body $body -ContentType "application/json"
```

#### ❌ Low Approval Probability Sample
```bash
curl -X POST "http://localhost:9090/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 22,
       "income": 25000,
       "education": 12,
       "experience": 1,
       "credit_score": 500
     }'
```

**PowerShell:**
```powershell
$body = '{"age":22,"income":25000,"education":12,"experience":1,"credit_score":500}'
Invoke-RestMethod -Uri "http://localhost:9090/predict" -Method POST -Body $body -ContentType "application/json"
```

#### 🔵 Medium Approval Probability Sample
```bash
curl -X POST "http://localhost:9090/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 28,
       "income": 45000,
       "education": 14,
       "experience": 5,
       "credit_score": 650
     }'
```

**PowerShell:**
```powershell
$body = '{"age":28,"income":45000,"education":14,"experience":5,"credit_score":650}'
Invoke-RestMethod -Uri "http://localhost:9090/predict" -Method POST -Body $body -ContentType "application/json"
```

---

## 📋 Alternative Input Formats

### MLflow Compatible Format
```bash
curl -X POST "http://localhost:9090/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "instances": [{
         "age": 35,
         "income": 75000,
         "education": 16,
         "experience": 10,
         "credit_score": 750
       }]
     }'
```

### Seldon Compatible Format  
```bash
curl -X POST "http://localhost:9090/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "ndarray": [[35, 75000, 16, 10, 750]]
       }
     }'
```

---

## 🔄 Expected Prediction Response

```json
{
  "predictions": [{
    "probability": 1.0,
    "prediction": "approved",
    "confidence": 1.0,
    "input": {
      "age": 35,
      "income": 75000,
      "education": 16,
      "experience": 10,
      "credit_score": 750
    }
  }],
  "model_name": "loan-approval-rf",
  "model_version": "1.0.0"
}
```

---

## 🛠️ Production Usage Tips

1. **Batch Predictions**: Send multiple requests in parallel
2. **Error Handling**: Always check HTTP status codes
3. **Monitoring**: Use the `/health` endpoint for monitoring
4. **Load Testing**: Test with realistic traffic volumes

---

## 🔗 Integration Examples

### Python Integration
```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:9090/predict",
    json={
        "age": 35,
        "income": 75000, 
        "education": 16,
        "experience": 10,
        "credit_score": 750
    }
)

result = response.json()
print(f"Prediction: {result['predictions'][0]['prediction']}")
```

### JavaScript Integration  
```javascript
const prediction = await fetch('http://localhost:9090/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    age: 35,
    income: 75000,
    education: 16, 
    experience: 10,
    credit_score: 750
  })
});

const result = await prediction.json();
console.log('Prediction:', result.predictions[0].prediction);
```

---

## 🎉 Success! Your Model is Production Ready!

✅ MLflow experiment tracking  
✅ MinIO artifact storage  
✅ Docker containerized deployment  
✅ REST API with multiple input formats  
✅ Health monitoring and metadata endpoints  
✅ Ready for integration with any application!