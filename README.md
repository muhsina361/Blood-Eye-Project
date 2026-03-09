
# Blood Eye Project

## Overview
The Blood Eye Project is an innovative web-based application that leverages advanced machine learning and computer vision techniques to predict human blood groups from eye images. This project explores the fascinating intersection of ophthalmology and hematology, where subtle characteristics in the eye's fundus (retina) and sclera (white outer layer) can provide insights into an individual's blood type.

### Scientific Background
Research has shown correlations between blood groups and various physiological traits, including vascular patterns in the eyes. The retina contains a network of blood vessels that can vary in density, tortuosity (curvature), and color based on blood composition. Similarly, the sclera may exhibit differences in redness, hue, and vessel visibility that correlate with blood types. By analyzing these features using deep learning and image processing algorithms, the system can classify blood groups (A, B, AB, O) with reasonable accuracy.

The application processes two types of eye images:
- **Fundus Images**: High-resolution photographs of the retina, capturing vascular networks and optic disc details.
- **Sclera Images**: Images focusing on the white part of the eye, revealing conjunctival vessels and tissue characteristics.

### Key Objectives
- Provide a non-invasive, image-based alternative to traditional blood typing methods.
- Demonstrate the potential of AI in medical diagnostics.
- Offer an educational tool for understanding machine learning applications in healthcare.

## Features
The application offers a comprehensive suite of features designed for both end-users and researchers:

### User Management
- **Registration and Login**: Secure user authentication system using SQLite database.
- **Session Management**: Maintains user sessions across page navigations.
- **Password Security**: Basic password storage (note: in production, implement hashing like bcrypt).

### Image Processing and Feature Extraction
- **Fundus Analysis**: Extracts 5 key features from retinal images:
  - CNN PCA1: Deep learning feature using ResNet50 and PCA dimensionality reduction.
  - AVR (Arteriovenous Ratio): Ratio of artery to vein widths in retinal vessels.
  - Vessel Redness: Average red channel intensity in vessel regions.
  - Tortuosity: Measure of vessel curvature and complexity.
  - Vessel Density: Percentage of image area occupied by blood vessels.
- **Sclera Analysis**: Extracts 5 complementary features:
  - CNN PCA1: Similar deep feature extraction.
  - AVR: Vessel diameter ratios in scleral vessels.
  - Mean Hue: Average color hue in scleral regions.
  - Redness: Red channel intensity in sclera.
  - Perivascular Contrast: Difference in intensity between vessels and surrounding tissue.

### Machine Learning Pipeline
- **Model Training**: Custom neural network training on eye feature datasets.
- **Prediction Engine**: Real-time blood group classification using trained models.
- **Model Persistence**: Saves trained models, scalers, and encoders for deployment.

### Visualization and Analytics
- **Training Metrics**: Accuracy and loss curves over training epochs.
- **Confusion Matrix**: Detailed classification performance visualization.
- **Feature Display**: Shows extracted features for uploaded images.

### Web Interface
- **Responsive Design**: Bootstrap-based UI for desktop and mobile access.
- **AJAX Uploads**: Asynchronous image uploads with progress feedback.
- **Results Dashboard**: Comprehensive display of predictions and features.

## Technologies Used
The project utilizes a modern Python-based tech stack optimized for machine learning and web development:

### Backend Framework
- **Flask**: Lightweight WSGI web framework for Python. Chosen for its simplicity, extensibility, and suitability for ML applications. Handles routing, session management, and template rendering.

### Machine Learning and Deep Learning
- **TensorFlow/Keras**: Primary framework for building and training neural networks. Provides high-level APIs for model construction, training, and inference. Used for both feature extraction (pre-trained ResNet50) and classification model.
- **Scikit-learn**: Comprehensive ML library for data preprocessing, model evaluation, and utility functions. Handles scaling, encoding, train-test splits, and metrics calculation.

### Computer Vision and Image Processing
- **OpenCV (cv2)**: Industry-standard computer vision library. Used for image loading, color space conversions, resizing, and basic manipulations.
- **Scikit-image**: Advanced image processing toolkit. Provides specialized filters like Frangi vessel enhancement, morphological operations, and region property analysis.

### Data Handling and Visualization
- **Pandas**: Powerful data manipulation library. Handles CSV dataset loading, cleaning, and feature engineering.
- **NumPy**: Fundamental package for scientific computing. Supports array operations, mathematical functions, and data transformations.
- **Matplotlib**: Plotting library for creating static, animated, and interactive visualizations. Generates training curves and performance graphs.
- **Seaborn**: Statistical data visualization based on Matplotlib. Creates attractive confusion matrices and statistical plots.

### Utilities and Storage
- **Joblib**: Efficient serialization of Python objects. Used for saving/loading ML models, scalers, and label encoders.
- **SQLite**: Embedded relational database. Manages user accounts and authentication data.
- **Werkzeug**: WSGI utility library. Provides secure filename handling and other Flask utilities.
- **UUID**: Generates unique identifiers for uploaded files to prevent conflicts.

### Frontend Technologies
- **HTML5/CSS3**: Semantic markup and styling for web pages.
- **Bootstrap**: Responsive CSS framework for modern UI components.
- **JavaScript/jQuery**: Client-side scripting for interactive elements and AJAX requests.

## Installation

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python Version**: 3.7 or higher (recommended: 3.8-3.10)
- **RAM**: Minimum 8GB, recommended 16GB for model training
- **Storage**: 5GB free space for models and datasets
- **GPU**: Optional but recommended for faster training (NVIDIA GPU with CUDA support)

### Prerequisites
1. **Python Installation**: Download from python.org and ensure it's added to PATH.
2. **Git**: For cloning repositories (optional).
3. **Virtual Environment**: Recommended for dependency isolation.

### Step-by-Step Installation Guide

#### 1. Download the Project
```bash
# Option 1: Clone from repository (if available)
git clone <repository-url>
cd "Blood Eye Project"

# Option 2: Download ZIP and extract to desired location
# Navigate to the extracted folder
```

#### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
```

#### 3. Install Dependencies
Choose one of the following methods:

**Method A: Using requirements.py (Recommended)**
```bash
cd Code
python requirements.py
```

**Method B: Manual Installation**
```bash
pip install flask werkzeug numpy pandas scikit-learn tensorflow matplotlib seaborn joblib opencv-python scikit-image pillow
```

**Method C: Using pip with requirements.txt (if available)**
```bash
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
python -c "import flask, tensorflow, cv2, sklearn; print('All dependencies installed successfully')"
```

### Troubleshooting Installation Issues

#### Common Problems and Solutions
- **TensorFlow Installation Issues**: If TensorFlow fails to install, try:
  ```bash
  pip install tensorflow-cpu  # CPU-only version
  ```
  Or for GPU support:
  ```bash
  pip install tensorflow[and-cuda]  # Requires compatible NVIDIA drivers
  ```

- **OpenCV Issues**: On some systems, install system dependencies:
  ```bash
  # Windows: Usually works out of the box
  # macOS: brew install opencv
  # Linux: sudo apt-get install libopencv-dev python3-opencv
  ```

- **Permission Errors**: Run terminal as administrator or use `--user` flag:
  ```bash
  pip install --user <package-name>
  ```

- **Virtual Environment Issues**: If activation fails, check execution policy (Windows):
  ```powershell
  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

## Usage

### Running the Application

#### Basic Startup
1. **Activate Environment**:
   ```bash
   cd "Blood Eye Project"
   venv\Scripts\activate  # Windows
   ```

2. **Navigate to Code Directory**:
   ```bash
   cd Code
   ```

3. **Start Flask Server**:
   ```bash
   python app.py
   ```

4. **Access Application**:
   - Open web browser
   - Navigate to: `http://127.0.0.1:5000/`
   - The application will automatically check for and train models if missing

#### First-Time Setup
On initial run, the application will:
- Create necessary directories (`uploads`, `static/uploads`)
- Check for model files (`trained_model.h5`, `scaler.pkl`, `encoder.pkl`)
- If models are missing, automatically run training using `eye_dataset2.csv`
- Display training progress in console

### User Workflow

#### 1. User Registration and Login
- **Access**: Visit homepage and click "Register" or "Login"
- **Registration**: Provide username and password
- **Login**: Enter credentials to access protected features
- **Security Note**: Passwords are stored in plain text; implement hashing for production

#### 2. Image Upload Process

**Fundus Image Upload**:
- Navigate to `/fundus_upload`
- Select and upload a retinal image (JPG/PNG format)
- System processes image and extracts features:
  - Loads image using OpenCV
  - Applies Frangi filter for vessel enhancement
  - Uses ResNet50 CNN for deep features
  - Calculates vessel metrics (density, redness, tortuosity)
  - Saves processed overlay image
- Returns JSON response with extracted features and processed image URL

**Sclera Image Upload**:
- Navigate to `/sclera_upload`
- Upload sclera image
- Feature extraction includes:
  - CNN feature extraction with PCA
  - Vessel segmentation and AVR calculation
  - Color analysis (hue, redness)
  - Perivascular contrast measurement
- Features stored in session for later use

#### 3. Blood Group Prediction
- **Access Results**: Visit `/results` after uploading both images
- **Feature Combination**: System combines 5 fundus + 5 sclera features
- **Preprocessing**: Applies trained scaler to normalize features
- **Prediction**: Feeds data to neural network for classification
- **Output**: Displays predicted blood group (A, B, AB, O) with confidence

#### 4. Model Training (Administrative Feature)
- **Access**: Login and visit `/train`
- **Upload Dataset**: Provide CSV file with eye features and blood group labels
- **Training Process**:
  - Loads and cleans data
  - Encodes categorical labels
  - Scales features using StandardScaler
  - Splits data (80% train, 20% test)
  - Trains neural network with early stopping
  - Saves model, scaler, and encoder
- **Visualization**: Generates and saves accuracy/loss plots and confusion matrix

#### 5. Analysis and Visualization
- **Access**: Visit `/analysis`
- **View Metrics**: Displays training performance graphs
- **Confusion Matrix**: Shows classification accuracy per blood group

### API Endpoints

The Flask application exposes several endpoints:

- `GET /`: Homepage
- `GET/POST /register`: User registration
- `GET/POST /login`: User authentication
- `GET /logout`: Session termination
- `GET /working`: Loading page
- `GET /chart`: Visualization page
- `GET/POST /fundus_upload`: Fundus image processing
- `GET/POST /sclera_upload`: Sclera image processing
- `GET /results`: Prediction results
- `GET /analysis`: Training analytics
- `GET/POST /train`: Model training interface

### Sample Data and Testing

#### Test Images
- Located in `Test Images/` directory
- Use these for testing upload and prediction features
- Ensure images are clear, well-lit eye photographs

#### Expected Input Formats
- **Image Formats**: JPG, PNG (recommended resolution: 1024x768 or higher)
- **Fundus Images**: Should show clear retinal vessels and optic disc
- **Sclera Images**: Focus on white scleral area with visible conjunctival vessels

## Dataset Description

### Dataset Structure
The project uses `eye_dataset2.csv` containing pre-extracted features from eye images:

#### Columns:
- **cnn_pca1**: First principal component of CNN features (float)
- **AVR**: Arteriovenous ratio (float)
- **vessel_redness**: Average red intensity in vessels (float)
- **sclera_mean_hue**: Mean hue in scleral regions (float)
- **AV_sat_diff**: Saturation difference between arteries and veins (float)
- **tortuosity**: Vessel curvature measure (float)
- **sclera_redness**: Red intensity in sclera (float)
- **vessel_density**: Percentage of vessel area (float)
- **perivascular_contrast**: Intensity contrast around vessels (float)
- **pulse_std**: Standard deviation of pulse-related features (float)
- **blood_group**: Target variable (categorical: A, B, AB, O)

### Data Collection
- Features extracted from fundus and sclera images using computer vision algorithms
- Ground truth blood groups obtained through traditional blood typing
- Dataset should be balanced across blood group classes for optimal training

### Data Preprocessing
- **Cleaning**: Remove rows with missing values
- **Encoding**: Convert blood groups to numerical labels
- **Scaling**: Standardize features to zero mean and unit variance
- **Splitting**: 80/20 train-test split with stratification

## Project Structure (Detailed)

```
Blood Eye Project/
├── Code/                          # Main application code
│   ├── app.py                     # Flask application with routes and logic
│   │   ├── Database setup and user management
│   │   ├── Image upload and processing routes
│   │   ├── Model prediction logic
│   │   └── Training integration
│   ├── training.py                # Standalone training script
│   │   ├── Data loading and preprocessing
│   │   ├── Model architecture definition
│   │   ├── Training loop with callbacks
│   │   └── Evaluation and visualization
│   ├── feature_Code_fundus.py     # Fundus feature extraction
│   │   ├── Image loading and preprocessing
│   │   ├── CNN feature extraction with ResNet50
│   │   ├── Vessel segmentation and analysis
│   │   └── Feature calculation and output
│   ├── feature_Code_scelera.py    # Sclera feature extraction
│   │   ├── Similar structure to fundus extraction
│   │   ├── Focus on scleral characteristics
│   │   └── Color and contrast analysis
│   ├── requirements.py            # Dependency management
│   ├── Training_Code.ipynb        # Jupyter notebook version
│   ├── *.h5 files                 # Keras model files
│   ├── *.pkl files                # Serialized scalers/encoders
│   └── templates/                 # Jinja2 HTML templates
│       ├── base.html              # Base template with navigation
│       ├── index.html             # Homepage
│       ├── login.html             # Authentication form
│       ├── register.html          # Registration form
│       ├── fundus_upload.html     # Fundus upload interface
│       ├── sclera_upload.html     # Sclera upload interface
│       ├── results.html           # Prediction display
│       ├── analysis.html          # Analytics dashboard
│       ├── train.html             # Training interface
│       └── working.html           # Loading screen
├── static/                        # Static web assets
│   ├── css/                       # Stylesheets
│   │   ├── main.css               # Custom styles
│   │   └── main1.css              # Additional styles
│   ├── js/                        # JavaScript files
│   │   ├── main.js                # Custom scripts
│   │   └── main1.js               # Additional scripts
│   ├── img/                       # Static images
│   ├── uploads/                   # User-uploaded files
│   └── vendor/                    # Third-party libraries
│       ├── bootstrap/             # CSS framework
│       ├── aos/                   # Animation library
│       ├── glightbox/             # Lightbox plugin
│       └── ...                    # Other vendor assets
├── Test Images/                   # Sample images for testing
├── Documents/                     # Project documentation
└── README.md                      # This documentation
```

## Model Details

### Neural Network Architecture
The classification model is a multi-layer perceptron implemented in Keras:

```
Input Layer: 10 features
├── Dense(128, activation='relu')
├── Dropout(0.3)
├── Dense(64, activation='relu')
├── Dropout(0.3)
├── Dense(32, activation='relu')
├── Dropout(0.2)
└── Dense(4, activation='softmax')  # 4 blood group classes
```

#### Layer Explanations:
- **Dense Layers**: Fully connected layers for feature learning
- **ReLU Activation**: Introduces non-linearity, prevents vanishing gradients
- **Dropout**: Regularization to prevent overfitting (rates: 0.3, 0.3, 0.2)
- **Softmax Output**: Probability distribution over blood group classes

### Training Configuration
- **Optimizer**: Adam (learning rate: 0.0005)
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy
- **Batch Size**: 16
- **Epochs**: Up to 150 (with early stopping)
- **Validation Split**: 20% of training data
- **Early Stopping**: Patience of 15 epochs, restore best weights

### Feature Engineering
- **Input Features**: 10-dimensional vector combining fundus and sclera features
- **Preprocessing**: Standard scaling to normalize feature distributions
- **Label Encoding**: Blood groups encoded as integers, then one-hot encoded

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Per-class performance analysis
- **Loss Curves**: Training and validation loss over epochs
- **Precision/Recall**: Calculated from confusion matrix

### Model Persistence
- **Model File**: `trained_model.h5` (HDF5 format)
- **Scaler**: `scaler.pkl` (StandardScaler object)
- **Encoder**: `encoder.pkl` (LabelEncoder object)
- **Loading**: Models loaded with `compile=False` for inference

## Feature Extraction Algorithms

### Fundus Feature Extraction
1. **Image Loading**: Read image, convert to RGB
2. **CNN Features**: 
   - Resize to 224x224
   - Preprocess for ResNet50
   - Extract features from avg pooling layer
   - Apply PCA to reduce to 1 component
3. **Vessel Enhancement**: Apply Frangi filter for vessel detection
4. **Vessel Metrics**:
   - Density: Binary mask area ratio
   - AVR: Mean vessel width ratio
   - Redness: Average red channel in vessels
   - Tortuosity: Curvature analysis using region properties

### Sclera Feature Extraction
1. **Image Processing**: Similar loading and CNN pipeline
2. **Vessel Segmentation**: Frangi filter with percentile thresholding
3. **Color Analysis**:
   - HSV conversion for hue extraction
   - Red channel analysis for redness
4. **Contrast Measurement**: Intensity differences around vessels

## Troubleshooting

### Common Issues and Solutions

#### Application Won't Start
- **Error**: "Module not found"
  - **Solution**: Ensure virtual environment is activated and dependencies installed
- **Error**: "Port already in use"
  - **Solution**: Kill process on port 5000 or change port in app.py

#### Model Training Fails
- **Error**: "Dataset not found"
  - **Solution**: Ensure `eye_dataset2.csv` exists in Code directory
- **Error**: "CUDA out of memory"
  - **Solution**: Reduce batch size or use CPU-only TensorFlow

#### Image Upload Issues
- **Error**: "Invalid file format"
  - **Solution**: Ensure JPG/PNG formats, check file corruption
- **Error**: "Feature extraction failed"
  - **Solution**: Verify image quality and content

#### Database Issues
- **Error**: "Database locked"
  - **Solution**: Close other connections or restart application

#### Performance Issues
- **Slow Training**: Use GPU, reduce epochs, or optimize architecture
- **High Memory Usage**: Process images in batches, use smaller models

### Debug Mode
Run with debug enabled for detailed error messages:
```bash
export FLASK_ENV=development
python app.py
```

### Logs and Monitoring
- Check console output for training progress and errors
- View browser developer tools for client-side issues
- Monitor system resources during training

## Future Enhancements

### Technical Improvements
- **Model Architecture**: Experiment with CNNs, transformers, or ensemble methods
- **Feature Engineering**: Add more sophisticated image features
- **Data Augmentation**: Implement image augmentation for better generalization
- **Transfer Learning**: Fine-tune larger pre-trained models

### User Experience
- **Batch Processing**: Allow multiple image uploads
- **Real-time Feedback**: Show processing progress
- **Mobile App**: Develop companion mobile application
- **API Access**: Provide REST API for external integrations

### Security and Privacy
- **Authentication**: Implement proper password hashing (bcrypt)
- **Data Privacy**: Add GDPR compliance features
- **Secure Uploads**: Validate and sanitize uploaded files
- **HTTPS**: Enable SSL/TLS encryption

### Scalability
- **Database**: Migrate to PostgreSQL or MongoDB
- **Deployment**: Containerize with Docker, deploy to cloud
- **Load Balancing**: Handle multiple concurrent users
- **Caching**: Implement Redis for session and result caching

### Research Directions
- **Accuracy Improvement**: Collect larger, more diverse datasets
- **Multi-modal Analysis**: Combine with other biometric data
- **Explainability**: Add model interpretation features
- **Clinical Validation**: Partner with medical institutions for validation

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Follow installation steps above
4. Make changes and test thoroughly
5. Commit with descriptive messages
6. Push to branch and create pull request

### Code Standards
- **Python Style**: Follow PEP8 guidelines
- **Documentation**: Add docstrings to functions
- **Testing**: Write unit tests for critical functions
- **Version Control**: Use meaningful commit messages

### Areas for Contribution
- **Bug Fixes**: Report and fix issues
- **Feature Requests**: Implement new functionalities
- **Documentation**: Improve README and code comments
- **Testing**: Add comprehensive test suites

## License
This project is released under the MIT License. See LICENSE file for details.

The MIT License allows:
- Commercial and private use
- Modification and distribution
- Private use without attribution
- Liability and warranty disclaimers apply

Note: Individual dependencies may have their own licenses. Check TensorFlow, scikit-learn, and other library licenses for commercial applications.

## References and Acknowledgments

### Scientific References
- Studies on retinal vessel analysis for cardiovascular risk assessment
- Research on ocular manifestations of blood groups
- Computer vision applications in ophthalmology

### Technical References
- TensorFlow/Keras Documentation
- Scikit-learn User Guide
- OpenCV Tutorials
- Flask Documentation

### Acknowledgments
- Open-source community for providing excellent libraries
- Research institutions for eye image datasets
- Contributors and testers

## Contact and Support

### Getting Help
- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Refer to this README and inline code comments

### Project Maintainers
- [Your Name/Organization]
- Email: [contact email]
- Repository: [GitHub URL]

### Version History
- **v1.0**: Initial release with basic functionality
- **Future versions**: Planned improvements and new features

---

*This README provides comprehensive documentation for the Blood Eye Project. For the latest updates, please check the repository.*
