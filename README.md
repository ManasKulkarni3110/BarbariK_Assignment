# FastAPI Data Visualization and Analytics API

A robust FastAPI application for data visualization and analytics with chat interface, real-time data processing, and advanced caching mechanisms.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-Latest-green.svg)

## 🚀 Features

- **Real-time Data Visualization**: Generate dynamic plots and charts from transaction data
- **Interactive Chat Interface**: Natural language processing for data queries
- **Advanced Caching**: Redis-based caching system for improved performance
- **Semantic Search**: Embedding-based search functionality using Sentence Transformers
- **Automated Plot Storage**: Automatic saving and management of generated visualizations
- **RESTful API**: Comprehensive API endpoints for data operations
- **Docker Support**: Containerized deployment with Docker and Docker Compose

## 📋 Prerequisites

- Python 3.9+
- Docker and Docker Compose
- MongoDB Atlas account (or local MongoDB instance)
- Redis
- Ollama

## 🛠️ Tech Stack

- **FastAPI**: Modern web framework for building APIs
- **MongoDB**: Document database for storing transaction data
- **Redis**: In-memory data store for caching
- **Matplotlib**: Data visualization
- **Sentence Transformers**: Natural language processing
- **Docker**: Containerization
- **Ollama**: ML model serving

## 🏗️ Project Structure

```
fastapi-app/
├── app.py                  # Main FastAPI application
├── plots/                  # Generated plot storage
├── logs/                   # Application logs
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
└── .env                    # Environment variables
```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/ManasKulkarni3110/BarbariK_Assignment
```

2. **Set up environment variables**
```bash
cp .env .env
# Edit .env with your configurations
```

3. **Build and run with Docker**
```bash
docker-compose up -d
```

4. **Initialize the database**
```bash
curl -X POST http://localhost:8000/init-database
```

## 🔌 API Endpoints

### Chat Interface
- `POST /chat`
  - Process natural language queries
  - Generate visualizations based on requests

### Data Operations
- `GET /transactions`
  - Retrieve transaction data
  - Filter by city, date range
- `POST /init-database`
  - Initialize sample data

### Visualization
- `GET /visualization`
  - Generate plots based on parameters
  - Supports multiple visualization types

### Testing
- `POST /stress-test`
  - Performance testing endpoint
  - Configurable load testing

## 📊 Visualization Examples

The API supports various visualization types:
- Time-series analysis
- City-wise comparisons
- Product performance metrics
- Customer behavior analysis

Example Query:
```bash
curl -X GET "http://localhost:8000/visualization?category=city&metric=sales_volume"
```

## 💾 Data Storage

### Plots
- Automatically saved in the `plots` directory
- Unique filename format: `{category}_{metric}_{timestamp}.png`
- High-resolution output (300 DPI)

### Caching
- Redis-based caching system
- Configurable cache expiration
- Improved response times for repeated queries

## 🔒 Security Features

- Environment variable configuration
- MongoDB authentication
- Rate limiting support
- Secure header configurations

## 🔧 Configuration

### Environment Variables
```env
MONGO_URI=mongodb://your_mongodb_uri
```

## 🚀 Deployment

### Local Development
```bash
# Run with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

## 📈 Performance Monitoring

- Built-in logging system
- Performance metrics tracking
- Stress testing capabilities

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## 👥 Authors

- Your Name - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- FastAPI documentation
- MongoDB documentation
- Redis documentation
- Ollama documentation

## 📞 Support

For support, email your.email@example.com or create an issue in the GitHub repository.

---
⭐️ Star this repository if you find it helpful!
