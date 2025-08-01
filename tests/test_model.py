import pytest
import tensorflow as tf
from src.models.train import build_model
import yaml

@pytest.fixture
def config():
    """Load test configuration"""
    with open('config/test_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_model_building(config):
    """Test that the model builds correctly"""
    model = build_model(config)
    assert model is not None
    assert isinstance(model, tf.keras.Model)
    
    # Check input/output shapes
    input_shape = tuple(config['data']['input_shape'])
    num_classes = config['data']['num_classes']
    assert model.input_shape[1:] == input_shape
    assert model.output_shape[-1] == num_classes

def test_model_training(config):
    """Test that the model can be trained on dummy data"""
    model = build_model(config)
    
    # Create dummy data
    input_shape = tuple(config['data']['input_shape'])
    num_samples = 10
    x_train = tf.random.normal((num_samples, *input_shape))
    y_train = tf.random.uniform((num_samples, 1), maxval=config['data']['num_classes'], dtype=tf.int32)
    
    # Train for 1 epoch
    history = model.fit(
        x_train, y_train,
        epochs=1,
        batch_size=config['data']['batch_size'],
        verbose=0
    )
    
    assert 'loss' in history.history
    assert 'accuracy' in history.history
