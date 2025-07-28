# OpenMed NER - AWS Marketplace Sample Notebooks

This repository contains sample notebooks and resources for using OpenMed NER models from AWS Marketplace.

## 📚 Available Models

### OpenMed NER Species Detection Model

- **Purpose**: Identify and extract species names from biomedical and clinical text
- **Entities**: Bacterial, fungal, viral, and other organism species
- **Use Cases**: Clinical analysis, research, infectious disease monitoring

## 🚀 Getting Started

### Prerequisites

- AWS account with SageMaker access
- Subscription to OpenMed models from AWS Marketplace
- Basic knowledge of Python and Jupyter notebooks

### Quick Start

1. Clone this repository
2. Open the relevant notebook in SageMaker Studio or Jupyter
3. Update the model package ARN with your marketplace subscription
4. Run the cells to deploy and test the model

## 📁 Repository Structure

```
aws-marketplace-notebooks/
├── OpenMed-NER-Species-Detection-Model.ipynb    # Main demonstration notebook
├── README.md                                     # This file
├── requirements.txt                              # Python dependencies
└── examples/                                     # Additional examples
    ├── batch_processing_example.py               # Batch processing script
    ├── clinical_text_samples.txt                 # Sample medical texts
    └── confidence_analysis.py                    # Confidence score analysis
```

## 🔧 Installation

### Option 1: SageMaker Studio

1. Upload the notebook to your SageMaker Studio environment
2. Install dependencies using the first cell in the notebook

### Option 2: Local Jupyter

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/aws-marketplace-notebooks.git
cd aws-marketplace-notebooks

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

## 📖 Usage Examples

### Basic Species Detection

```python
# Import required libraries
import boto3
import sagemaker
from sagemaker import ModelPackage

# Configure your model
model_package_arn = "arn:aws:sagemaker:us-east-1:YOUR-ACCOUNT:model-package/YOUR-MODEL"
model = ModelPackage(model_package_arn=model_package_arn, role=role)

# Deploy and predict
predictor = model.deploy(instance_type="ml.m5.large")
result = predictor.predict({"inputs": "Patient infected with Streptococcus pneumoniae"})
```

### Expected Output Format

```json
[
  {
    "entity_group": "SPECIES",
    "score": 0.9823,
    "word": "Streptococcus pneumoniae",
    "start": 20,
    "end": 44
  }
]
```

## 💡 Best Practices

### Input Optimization

- **Text Length**: Keep input under 512 tokens for optimal performance
- **Text Quality**: Use clean, well-formatted medical text
- **Language**: Model optimized for English medical content

### Confidence Thresholds

- **High Precision** (>0.9): Critical clinical applications
- **Balanced** (>0.8): General medical analysis
- **High Recall** (>0.7): Exploratory research

### Performance Tips

- Use `ml.m5.large` or larger instances for production
- Implement batch processing for large datasets
- Cache results for repeated analyses
- Monitor endpoint costs and usage

## 🏥 Medical Use Cases

### Clinical Applications

- **Infectious Disease Monitoring**: Track pathogen species in clinical reports
- **Antimicrobial Stewardship**: Identify resistant organisms
- **Epidemiology**: Analyze disease patterns and outbreaks

### Research Applications

- **Literature Mining**: Extract species from research papers
- **Microbiome Studies**: Analyze microbial community compositions
- **Drug Discovery**: Identify target organisms and interactions

## 🔒 Security and Compliance

- All data processing occurs within your AWS account
- No data is sent to external services
- HIPAA-eligible when deployed in compliant AWS environments
- Supports VPC deployment for enhanced security

## 💰 Cost Optimization

### Instance Recommendations

- **Development**: `ml.t2.medium` ($0.06/hour)
- **Production**: `ml.m5.large` ($0.12/hour)
- **High Performance**: `ml.m5.xlarge` ($0.24/hour)

### Cost-Saving Tips

- Use SageMaker Serverless Inference for intermittent usage
- Implement auto-scaling for variable workloads
- Delete endpoints when not in use
- Consider batch transform for large-scale processing

## 🔧 Troubleshooting

### Common Issues

**Deployment Errors**

- Verify model package ARN is correct
- Check IAM permissions for SageMaker
- Ensure sufficient service quotas

**Prediction Errors**

- Validate JSON input format
- Check text encoding (UTF-8)
- Verify endpoint is in "InService" status

**Performance Issues**

- Scale to larger instance types
- Optimize input text preprocessing
- Implement request batching

### Support Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [AWS Marketplace Support](https://aws.amazon.com/marketplace/help/)
- Model-specific support via AWS Marketplace listing

## 🚀 Advanced Features

### Batch Processing

Process large datasets efficiently using SageMaker Batch Transform:

```python
# Create batch transform job
transformer = model.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://your-bucket/output/'
)

transformer.transform(
    data='s3://your-bucket/input/',
    content_type='application/json'
)
```

### Multi-Model Endpoints

Deploy multiple OpenMed models on a single endpoint for cost optimization.

### Custom Preprocessing

Implement domain-specific text preprocessing for enhanced accuracy.

## 📈 Monitoring and Analytics

### CloudWatch Metrics

- Endpoint invocations
- Model latency
- Error rates
- Instance utilization

### Custom Monitoring

```python
# Track prediction confidence
import boto3

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_data(
    Namespace='OpenMed/NER',
    MetricData=[
        {
            'MetricName': 'PredictionConfidence',
            'Value': confidence_score,
            'Unit': 'None'
        }
    ]
)
```

## 🤝 Contributing

We welcome contributions to improve these notebooks and examples!

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests and documentation
5. Submit a pull request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Include docstrings for functions
- Add appropriate error handling
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Changelog

### Version 1.0.0 (2025-01-XX)

- Initial release
- OpenMed NER Species Detection notebook
- Basic usage examples
- Performance optimization guides

## 📞 Support

For technical support and questions:

1. **AWS Marketplace**: Contact through your marketplace subscription
2. **GitHub Issues**: Report bugs and feature requests
3. **AWS Forums**: General SageMaker questions
4. **Documentation**: Comprehensive guides and examples

---

**Model Provider**: OpenMed AI
**AWS Marketplace**: [View Listing](https://aws.amazon.com/marketplace/)
**Documentation**: [Complete Guide](https://docs.aws.amazon.com/sagemaker/)

## ⭐ Star this Repository

If you find these notebooks helpful, please consider starring this repository to help others discover it!

---

*Last Updated: July 2025*
